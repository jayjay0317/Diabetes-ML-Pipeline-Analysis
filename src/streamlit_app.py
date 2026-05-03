import streamlit as st
import pandas as pd
from model_handler import DiabetesPredictor

# Page configuration
st.set_page_config(page_title='Diabetes Risk Predictor', layout='centered')

# Load model with caching to prevent redundant loading
@st.cache_resource
def get_predictor():
    predictor = DiabetesPredictor()
    predictor.load_model('notebooks/diabetes_rf_model.pkl')
    return predictor

predictor = get_predictor()

# Header section
st.title('🏥 Diabetes Risk Predictor')
st.write('Please enter your health indicators to check the diabetes risk level.')

# Age category mapping for better UX
age_labels = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+"
}

# Input form for user data
with st.form('diabetes_form'):
    st.subheader('User Health Indicators')

    # Input widgets for key features
    bmi = st.slider('BMI (Body Mass Index)', 10.0, 60.0, 25.0)
    age = st.select_slider(
        'Age Category',
        options=list(age_labels.keys()), value=5,
        format_func=lambda x: age_labels[x]
    )
    high_bp = st.selectbox(
        'High Blood Pressure',
        [0, 1],
        format_func=lambda x: 'Yes' if x==1 else 'No'
    )

    # Form submission button
    submitted = st.form_submit_button('Predict Result')

# Inference logic upon form submission
if submitted:
    # Feature engineering: matching the 21 input features required by the model
    # Placeholders (0) are used for the remaining 18 features for now
    features = [high_bp, 0, 0, bmi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, age, 0, 0]

    # Perform prediction (returns the first element directly)
    prediction = predictor.predict(features)

    # Display results with visual feedback
    if prediction == 1:
        st.error('⚠️ High Risk: Clinical consultation is recommended.')
    else:
        st.success("✅ Low Risk: Maintain your healthy lifestyle!")