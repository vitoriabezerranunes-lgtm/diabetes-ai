import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="DiabetesAI — Risk Prediction",
    layout="centered",
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .low-risk { background-color: #dcfce7; color: #166534; border: 2px solid #16a34a; }
    .high-risk { background-color: #fee2e2; color: #991b1b; border: 2px solid #dc2626; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("DiabetesAI")
st.subheader("Diabetes Risk Prediction")
st.markdown(
    "Enter your health data below to estimate your diabetes risk. "
    "**This result is informational and does not replace medical advice.**"
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration 2 hours after an oral glucose tolerance test."
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index: weight divided by height squared. Normal range: 18.5–24.9."
    )
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=30,
        help="Age in full years."
    )
    blood_pressure = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure (lower number). Normal: below 80 mmHg."
    )

with col2:
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=20, value=0,
        help="How many times the patient has been pregnant."
    )
    insulin = st.number_input(
        "Insulin (µU/mL)",
        min_value=0, max_value=900, value=80,
        help="2-hour serum insulin level. Normal fasting range: 2–25 µU/mL."
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness in millimeters. Indicates body fat reserve."
    )
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="Score estimating diabetes likelihood based on family history. Typical range: 0.08–2.42."
    )

st.divider()

if st.button("Calculate Diabetes Risk"):
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])

    prob = model.predict_proba(input_data)[0][1]
    percent = prob * 100

    if prob < 0.5:
        st.markdown(
            f'<div class="result-box low-risk">Low Risk: {percent:.1f}%</div>',
            unsafe_allow_html=True,
        )
        st.success(
            "Your data indicates **low diabetes risk**. Keep up healthy habits: "
            "balanced diet, regular physical activity, and routine medical check-ups."
        )
    else:
        st.markdown(
            f'<div class="result-box high-risk">High Risk: {percent:.1f}%</div>',
            unsafe_allow_html=True,
        )
        st.error(
            "Your data indicates **high diabetes risk**. Please consult a doctor "
            "for confirmatory tests and professional guidance."
        )

    with st.expander("View analysis details"):
        st.write(f"- Probability of **not having** diabetes: **{(1-prob)*100:.1f}%**")
        st.write(f"- Probability of **having** diabetes: **{percent:.1f}%**")
        st.write("- Model: XGBoost trained on the Pima Indians Diabetes dataset")

st.divider()
st.caption(
    "DiabetesAI · Built with XGBoost and Streamlit · "
    "Dataset: Pima Indians Diabetes (UCI / Kaggle)"
)
