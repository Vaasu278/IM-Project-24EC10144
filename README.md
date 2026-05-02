# IM-Project-24EC10144

# Predictive Maintenance Dashboard

A production-ready Streamlit application that uses **Large Language Models (LLM)** to extract sensor data from unstructured operator logs and **Machine Learning (Random Forest)** to predict machine defects in real-time.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=googlegemini&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikitlearn&logoColor=white)

---

## Key Features

- **Unstructured Data Extraction**: Paste raw maintenance notes (e.g., "The machine was running hot at 305K...") and the Gemini LLM automatically extracts specific numerical sensor values.
- **Real-Time Prediction**: Instantly classifies machine health as **Normal** or **Defect Detected** using a Random Forest model.
- **Premium UI/UX**: Custom dark-grey theme with glassmorphism effects, live sensor cards, and flashing alerts for critical defects.
- **Robust Pipeline**: Includes error handling for API failures, parsing errors, and missing model files.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM API**: [Google Gemini 2.5 Flash](https://aistudio.google.com/app/apikey)
- **Data Science**: Scikit-Learn, Pandas, NumPy
- **Styling**: Vanilla CSS + Streamlit Themes

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will also need a **Gemini API Key** (Free) from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Installation
Clone the repository and install the dependencies:

```bash
pip install streamlit google-genai numpy pandas scikit-learn category_encoders imbalanced-learn
```

### 3. Running the App
Launch the dashboard from your terminal:

```bash
streamlit run app.py
```

---

## Project Structure

```text
archive/
├── app.py                # Main Streamlit application logic
├── model.pkl             # Trained Random Forest model (6 features)
├── predictive_maintenance.csv # Raw dataset
├── model_train.ipynb        # Model training and EDA notebook
└── .streamlit/
    └── config.toml       # Custom dark theme configuration
```

---

## Model & Pipeline Details

### The Extraction Pipeline
1. **Input**: User pastes an operator log.
2. **LLM**: Gemini 2.5 Flash processes the text with a specialized system prompt.
3. **Parser**: A robust Python parser (AST + Regex) converts the LLM's string response into a list of 5 floats.
4. **Features**: The pipeline constructs a 6-feature array: `[Machine_Type, Air_Temp, Process_Temp, Speed, Torque, Tool_Wear]`.

### Training Strategy
The model was trained on the **Predictive Maintenance Dataset**. 
- **Leakage Fix**: We explicitly removed the `Target` column from the feature set to ensure real-world validity.
- **Class Imbalance**: Used **SMOTETomek** to handle the minority "Defect" class, improving recall from near-zero to ~79%.

---

## Example Logs to Try
**Machine Type to be selected from the sidebar**

**Machine Type: L (Low)**
> "Shift report for Machine Unit 7. Air temperature logged at 298.8 K. Process temperature 308.9 K. Rotational speed was 1455 rpm, torque 41.3 Nm. Cumulative tool wear has reached 208 minutes."
> **Result**: Defect Detected (Tool Wear Failure)

**Machine Type: M (Medium)**
> "Unit running normally. Air temp 298.1 K, process temp 308.6 K. Speed 1551 rpm, torque 42.8 Nm. Wear at 0 min."
> **Result**: Normal Operation

---
