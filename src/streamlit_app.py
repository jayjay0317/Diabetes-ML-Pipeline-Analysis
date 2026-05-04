import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
    
    st.markdown('---')
    st.subheader('Lifestyle & Socioeconomic Factors')

    col3, col4 = st.columns(2)

    with col3:
        smoker = st.radio(
            'Smoker', ['No', 'Yes'],
            help='Have you smoked at least 100 cigarettes in your life?'
            )
        ment_hlth = st.slider(
            'Mental Health (Poor Days)', 0, 30, 0,
            help='Days in past 30 days your mental health was not good')
    
    with col4:
        # Map numeric categories to descriptive labels
        edu_labels = {
            1: 'Never attended school', 2: 'Grades 1-8 (Elementary)',
            3: 'Grades 9-11 (Some high school)', 4: 'High school graduate', 
            5: 'Some college or technical school', 6: 'College graduate'
        }
        education = st.selectbox(
            'Education Level',
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: edu_labels[x],
            index=5 # Default to College graduate
        )

        inc_labels = {
            1: 'Less than $10,000', 2: '$10K - $15K', 3: '$15K - $20K', 
            4: '$20K - $25K', 5: '$25K - $35K', 6: '$35K - $50K', 
            7: '$50K - $75K', 8: '$75,000 or more'
        }
        income = st.selectbox(
            'Income Level',
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: inc_labels[x],
            index=7 # Default to highest income bracket
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
    input_df.at[0, 'Smoker'] = 1.0 if smoker == "Yes" else 0.0
    input_df.at[0, 'MentHlth'] = float(ment_hlth)
    input_df.at[0, 'Education'] = float(education)
    input_df.at[0, 'Income'] = float(income)

    # Perform prediction (returns the first element directly)
    prediction = predictor.predict(input_df)

    # Display results with visual feedback
    if prediction == 1:
        st.error('⚠️ High Risk: Clinical consultation is recommended.')
    else:
        st.success('✅ Low Risk: Maintain your healthy lifestyle!')

    # --- Feature Importance Visualization ---
    st.markdown('---')
    st.subheader('📊 What factors influenced your risk?')

    # Extract top 10 features and convert Series to DataFrame for Altair compatibility
    importances = predictor.get_feature_importance().head(10)
    importance_df = importances.reset_index()
    importance_df.columns = ['Feature', 'Importance']

    # Render bar chart with explicit sorting to prevent alphabetical order
    chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('Feature', sort='-y', title='Health Factors'),
        y=alt.Y('Importance', title='Impact Weight'),
        color=alt.Color('Importance', scale=alt.Scale(scheme='blues'), legend=None)
    )

    st.altair_chart(chart, use_container_width=True)

    st.info("""
**Top Risk Drivers:** The chart above shows which health factors the AI prioritized 
when calculating your specific result. Factors like GenHlth, HighBP, and BMI 
typically play the largest roles in this model's decision-making.
""")