import streamlit as st
import pandas as pd
import numpy as np
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

# Map numeric age categories to human-readable strings
age_labels = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+"
}

# Map numeric general health categories to human-readable stgrings
genhlth_labels = {
    1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair',
    5: 'Poor'
}

# Input form for user data
with st.form('diabetes_form'):
    st.subheader('User Health Indicators')

    # Input widgets for key features
    bmi = st.slider('BMI (Body Mass Index)', 10.0, 60.0, 25.0)
    age = st.select_slider(
        'Age Range',
        options=list(age_labels.keys()), 
        value=5,
        format_func=lambda x: age_labels[x]
    )
    genhlth = st.select_slider(
        'General Health',
        options=list(genhlth_labels.keys()),
        value=3,
        format_func=lambda x: genhlth_labels[x]
    )
    high_bp = st.selectbox(
        'High Blood Pressure',
        [0, 1],
        format_func=lambda x: 'Yes' if x==1 else 'No'
    )
    st.markdown('---')
    st.subheader('Additional Health Indicators')

    col1, col2 = st.columns(2)
    with col1:
        high_chol = st.radio(
            'High Cholesterol', ['No', 'Yes'],
            help='Has a doctor told you that your cholesterol is high?'
            )
        phys_activity = st.radio(
            'Physical Activity', ['Yes', 'No'],
            help='Any physical activity or exercise in the past 30 days?'
            )
        
    with col2:
        diff_walk = st.radio(
            'Difficulty Walking', ['No', 'Yes'], 
            help='Do you have serious difficulty walking or climbing stairs?'
            )

    # Form submission button
    submitted = st.form_submit_button('Predict Result')

# Inference logic upon form submission
if submitted:
    # Feature engineering: matching the 21 input features required by the model
    # Placeholders (0) are used for the remaining 18 features for now
    # Define exact column names used during training
    column_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]

    # Creat a DataFrame to ensure the pipeline identifies features by name
    input_df = pd.DataFrame([[0.0] * 21], columns=column_names)

    input_df.at[0, 'HighBP'] = float(high_bp)
    input_df.at[0, 'BMI'] = float(bmi) # Pipeline will apply log1p and scaling
    input_df.at[0, 'Age'] = float(age) # Pipeline will apply scaling
    input_df.at[0, 'GenHlth'] = float(genhlth)
    input_df.at[0, 'HighChol'] = 1.0 if high_chol == 'Yes' else 0.0
    input_df.at[0, 'PhysActivity'] = 1.0 if phys_activity == 'Yes' else 0.0
    input_df.at[0, 'DiffWalk'] = 1.0 if diff_walk == 'Yes' else 0.0
     
    # Perform prediction (returns the first element directly)
    prediction = predictor.predict(input_df)

    # Display results with visual feedback
    if prediction == 1:
        st.error('⚠️ High Risk: Clinical consultation is recommended.')
    else:
        st.success('✅ Low Risk: Maintain your healthy lifestyle!')