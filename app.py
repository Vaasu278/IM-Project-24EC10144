"""
Predictive Maintenance Dashboard
---------------------------------
Pipeline:
  1. Operator pastes a free-text maintenance log.
  2. Gemini LLM extracts 5 sensor readings as a Python list.
  3. Robust parser (ast + regex fallback) validates the list.
  4. Numpy array is fed to a pre-trained model.pkl (sklearn).
  5. Prediction is displayed: green = Normal, red = Defect Detected.
"""

import ast
import re
import os

import numpy as np
import streamlit as st
from google import genai
from google.genai import types as genai_types
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration & global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom CSS for the flash animation and extra polish
st.markdown(
    """
    <style>
        /* ── General layout ─────────────────────────────── */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* ── Section divider ────────────────────────────── */
        .section-divider {
            border: none;
            border-top: 1px solid #33333A;
            margin: 1.5rem 0;
        }

        /* ── Sensor card grid ───────────────────────────── */
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .sensor-card {
            background: #26262C;
            border: 1px solid #3A3A44;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .sensor-label {
            font-size: 0.72rem;
            color: #9090A8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .sensor-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #4F8EF7;
        }

        /* ── Normal result box ──────────────────────────── */
        .result-normal {
            background: linear-gradient(135deg, #0d2b1e, #113d29);
            border: 1.5px solid #27ae60;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
        }
        .result-normal h2 { color: #2ecc71; margin: 0; font-size: 2rem; }
        .result-normal p  { color: #a0d8b8; margin: 0.4rem 0 0; font-size: 1rem; }

        /* ── Defect result box with flash animation ─────── */
        @keyframes flash-border {
            0%   { border-color: #e74c3c; box-shadow: 0 0 0px #e74c3c; }
            50%  { border-color: #ff6b6b; box-shadow: 0 0 18px #e74c3c; }
            100% { border-color: #e74c3c; box-shadow: 0 0 0px #e74c3c; }
        }
        .result-defect {
            background: linear-gradient(135deg, #2b0d0d, #3d1111);
            border: 2px solid #e74c3c;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
            animation: flash-border 1.2s ease-in-out infinite;
        }
        .result-defect h2 { color: #e74c3c; margin: 0; font-size: 2rem; }
        .result-defect p  { color: #d8a0a0; margin: 0.4rem 0 0; font-size: 1rem; }

        /* ── Stale footer ───────────────────────────────── */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

SENSOR_LABELS = [
    "Air Temp (K)",
    "Process Temp (K)",
    "Rot. Speed (rpm)",
    "Torque (Nm)",
    "Tool Wear (min)",
]

# Ordinal encoding used by ce.OrdinalEncoder in training: L=1, M=2, H=3
TYPE_ENCODING = {"L (Low)": 1, "M (Medium)": 2, "H (High)": 3}

SYSTEM_PROMPT = (
    "You are a data extraction tool. Extract the following 5 variables from the text: "
    "Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear. "
    "Return the data EXACTLY as a Python list of 5 floats in that exact order: "
    "[Air_Temp, Process_Temp, Speed, Torque, Wear]. "
    "Do not include any other text, markdown formatting, or explanations. "
    "If a value is not mentioned, insert 0.0."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: robust LLM output → Python list parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_llm_list(raw: str) -> list[float]:
    """
    Robustly extract a Python list of 5 floats from an LLM response string.
    Strategy:
      1. Strip markdown fences / stray backticks.
      2. Try ast.literal_eval on the cleaned string.
      3. Fall back to a regex that grabs the bracketed content.
    Raises ValueError if neither strategy yields a 5-element list.
    """
    # Step 1 – strip code fences and backticks
    cleaned = re.sub(r"```[a-zA-Z]*", "", raw).replace("`", "").strip()

    # Step 2 – attempt direct eval
    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, list) and len(result) == 5:
            return [float(v) for v in result]
    except (ValueError, SyntaxError):
        pass

    # Step 3 – regex: find first [...] block
    match = re.search(r"\[([^\[\]]+)\]", cleaned)
    if match:
        inner = match.group(1)
        try:
            values = [float(x.strip()) for x in inner.split(",")]
            if len(values) == 5:
                return values
        except ValueError:
            pass

    raise ValueError(
        f"Could not parse a list of 5 floats from LLM output:\n{raw!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: call Gemini API
# ─────────────────────────────────────────────────────────────────────────────
def extract_sensors_via_llm(log_text: str, api_key: str) -> str:
    """Send the operator log to Gemini and return the raw response string."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=log_text,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=512,  # raised – thinking models need more budget
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=0,  # disable thinking; not needed for simple extraction
            ),
        ),
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model (cached so it is only loaded once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# UI – Header
# ─────────────────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([0.05, 0.95])
with col_icon:
    st.markdown("## ⚙️")
with col_title:
    st.markdown("## Predictive Maintenance Dashboard")
    st.caption("Paste an operator log → LLM extracts sensor data → ML model predicts machine health.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI – Sidebar: API key input
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at aistudio.google.com/app/apikey. Used only for this session.",
    )
    st.markdown(
        "🔗 [Get a free Gemini API key](https://aistudio.google.com/app/apikey)",
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.markdown("### ⚙️ Machine Settings")
    machine_type = st.selectbox(
        "Machine Type",
        options=list(TYPE_ENCODING.keys()),
        index=1,
        help="Select the machine quality type (Low / Medium / High). This matches the 'Type' column in the training data.",
    )
    st.markdown("---")
    st.markdown(
        "**Model file:** `model.pkl`  \n"
        "Place `model.pkl` in the same directory as `app.py`."
    )
    st.markdown(
        "**Expected features** (in order):  \n"
        "1. Air Temperature (K)  \n"
        "2. Process Temperature (K)  \n"
        "3. Rotational Speed (rpm)  \n"
        "4. Torque (Nm)  \n"
        "5. Tool Wear (min)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# UI – Main area: log input
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([0.6, 0.4], gap="large")

with left_col:
    st.markdown("#### 📋 Operator Maintenance Log")
    log_text = st.text_area(
        label="Enter the operator log below:",
        placeholder=(
            "e.g.  Machine #7 – Shift 2 report.\n"
            "Air temperature measured at 298.5 K, process temperature 309.2 K.\n"
            "Rotational speed was steady at 1500 rpm. Torque peaked at 42.8 Nm.\n"
            "Tool wear accumulated over 215 minutes. No other anomalies observed."
        ),
        height=260,
        label_visibility="collapsed",
    )

    analyze_btn = st.button(
        "🔍 Analyze Log",
        type="primary",
        use_container_width=True,
        disabled=(not log_text.strip()),
    )

with right_col:
    st.markdown("#### 💡 Tips for Accurate Results")
    st.info(
        "Include all five sensor readings in your log for best accuracy:\n\n"
        "- **Air Temperature** – ambient temp in Kelvin\n"
        "- **Process Temperature** – coolant/process temp in Kelvin\n"
        "- **Rotational Speed** – shaft speed in rpm\n"
        "- **Torque** – measured torque in Nm\n"
        "- **Tool Wear** – cumulative wear in minutes\n\n"
        "Missing values will default to **0.0**."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    # ── Guard: API key ───────────────────────────────────────────────────────
    api_key = api_key_input.strip() or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error(
            "⚠️ No Gemini API key provided. Enter it in the sidebar or set "
            "the **GEMINI_API_KEY** environment variable.  \n"
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
        st.stop()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 1: LLM extraction ───────────────────────────────────────────────
    with st.spinner("🤖 Sending log to Gemini for extraction…"):
        try:
            raw_llm_output = extract_sensors_via_llm(log_text, api_key)
        except Exception as exc:
            st.error(f"❌ OpenAI API error: {exc}")
            st.stop()

    with st.expander("🔎 Raw LLM response (debug)", expanded=False):
        st.code(raw_llm_output, language="text")

    # ── Step 2: Parse LLM output ─────────────────────────────────────────────
    try:
        sensor_values = parse_llm_list(raw_llm_output)
    except ValueError as exc:
        st.error(f"❌ Parsing error – {exc}")
        st.stop()

    # ── Step 3: Display extracted sensor array ───────────────────────────────
    st.markdown("#### 📡 Extracted Sensor Readings")

    # Show machine type chip
    type_code = TYPE_ENCODING[machine_type]
    st.caption(f"🏭 Machine Type: **{machine_type}** (encoded as `{type_code}`)")

    cards_html = '<div class="sensor-grid">'
    for label, value in zip(SENSOR_LABELS, sensor_values):
        cards_html += (
            f'<div class="sensor-card">'
            f'<div class="sensor-label">{label}</div>'
            f'<div class="sensor-value">{value:,.2f}</div>'
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Build 6-feature array: [Type, Air, Process, Speed, Torque, Wear]
    full_features = [type_code] + sensor_values
    feature_array = np.array(full_features, dtype=float).reshape(1, -1)
    st.caption(f"NumPy array passed to model (6 features): `{feature_array.tolist()}`")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 4: Model prediction ─────────────────────────────────────────────
    with st.spinner("⚙️ Running model inference…"):
        try:
            model = load_model(MODEL_PATH)
            prediction = int(model.predict(feature_array)[0])
        except FileNotFoundError:
            st.error(
                f"❌ Model file not found at `{MODEL_PATH}`. "
                "Place `model.pkl` in the same directory as `app.py` and restart."
            )
            st.stop()
        except Exception as exc:
            st.error(f"❌ Model prediction error: {exc}")
            st.stop()

    # ── Step 5: Display result ───────────────────────────────────────────────
    st.markdown("#### 🏁 Prediction Result")

    if prediction == 0:
        st.markdown(
            '<div class="result-normal">'
            "<h2>✅ Normal Operation</h2>"
            "<p>The model predicts no defect. Machine health is within expected parameters.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-defect">'
            "<h2>🚨 Defect Detected!</h2>"
            "<p>The model has flagged a potential machine defect. "
            "Schedule immediate maintenance and inspection.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Optionally show prediction probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feature_array)[0]
            st.markdown("")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("P(Normal)", f"{proba[0]*100:.1f}%")
            with prob_col2:
                st.metric("P(Defect)", f"{proba[1]*100:.1f}%")
        except Exception:
            pass  # silently skip if proba not available
