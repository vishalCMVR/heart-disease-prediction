import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    height: 3.2em;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    font-size: 18px;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #ff6b6b, #ff4b4b);
}

.card {
    padding: 25px;
    border-radius: 18px;
    background-color: #161B22;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #A0A0A0;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI-powered cardiovascular risk analysis using Logistic Regression</div>',
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 20, 100, 40)

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.slider(
        "Resting Blood Pressure",
        80,
        200,
        120
    )

    chol = st.slider(
        "Cholesterol Level",
        100,
        600,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    restecg = st.selectbox(
        "Rest ECG Results",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

with col2:

    thalach = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        0.0,
        6.0,
        1.0
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.selectbox(
        "Major Vessels Colored",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )

# Mapping strings to model values
sex_map = {"Male": 1, "Female": 0}

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

fbs_map = {"No": 0, "Yes": 1}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang_map = {"No": 0, "Yes": 1}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# Predict Button
if st.button("Analyze Heart Disease Risk"):

    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex_map[sex],
        'cp': cp_map[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_map[fbs],
        'restecg': restecg_map[restecg],
        'thalach': thalach,
        'exang': exang_map[exang],
        'oldpeak': oldpeak,
        'slope': slope_map[slope],
        'ca': ca,
        'thal': thal_map[thal]
    }])

    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.35

    st.markdown("---")

    st.subheader(f"Predicted Risk Score: {probability:.2%}")

    if probability >= threshold:

        st.error(
            "⚠️ High Risk of Heart Disease Detected"
        )

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )