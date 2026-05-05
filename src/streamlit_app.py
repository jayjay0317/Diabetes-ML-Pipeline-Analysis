import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from model_handler import DiabetesPredictor

# Page configuration
st.set_page_config(page_title='Diabetes Risk Predictor', layout='centered')

# Load model with caching to prevent redundant loading
@st.cache_resource
def get_predictor_v2():
    predictor = DiabetesPredictor()
    predictor.load_model('notebooks/diabetes_rf_model.pkl')
    return predictor

def map_age_to_category(age):
    """
    Map raw age to the 13-level category used in the BRFSS dataset.
    """
    if age < 25:
        return 1.0
    elif age < 30:
        return 2.0
    elif age < 35:
        return 3.0
    elif age < 40:
        return 4.0
    elif age < 45:
        return 5.0
    elif age < 50:
        return 6.0
    elif age < 55:
        return 7.0
    elif age < 60:
        return 8.0
    elif age < 65:
        return 9.0
    elif age < 70:
        return 10.0
    elif age < 75:
        return 11.0
    elif age < 80:
        return 12.0
    else:
        return 13.0

predictor = get_predictor_v2()

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
    col_h, col_w = st.columns(2)
    with col_h:
        height_cm = st.number_input(
            'Height (cm)', min_value=100.0, max_value=250.0, value=None,
            step=1.0, format='%.1f', help='Enter your height in centimeters',
            placeholder='e.g. 170.5'
            )
    with col_w:
        weight_kg = st.number_input(
            'Weight (kg)', min_value=30.0, max_value=200.0, value=None,
            step=1.0, format='%.1f', help='Enter your weight in kilograms',
            placeholder='e.g. 60.3'
            )

    age_input = st.number_input(
        'Age', min_value=18, max_value=120, value=None,
        step=1, placeholder='e.g. 35'
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
    # Calculate BMI
    if height_cm is None or weight_kg is None:
        st.warning('Please enter both your height and weight to proceed.')
        st.stop()   

    if age_input is None:
        st.warning('Please enter your age.')
        st.stop()

    bmi = weight_kg / ((height_cm / 100) ** 2)
    age = map_age_to_category(age_input)
   
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

    # Extract probability of the positive class (High Risk)
    probabilities = predictor.predict_proba(input_df)
    high_risk_prob = probabilities[1]

    # Apply optimal threshold derived from CV Youden's J statistic
    optimal_threshold = 0.419

    # Display prediction results based on the custom threshold
    st.markdown('---')
    st.subheader('🩺 Screening Result')

    # Display calculated BMI
    st.info(f'⚖️ Based on your height and weight, your calculated BMI is **{bmi:.1f}**.')

    # Convert probability to a 100 point Risk Score
    risk_score = high_risk_prob * 100
    threshold_score = optimal_threshold * 100

    # Display prediction results based on the custom threshold
    if high_risk_prob >= optimal_threshold:
        st.error(f'⚠️ **High Risk** (Risk Score: {risk_score:.1f} / 100)')
        st.progress(high_risk_prob) # Visual indicator
        st.write('Clinical consultation is recommended based on this screening.')

        # Add an expandable explanation for users wondering about the score
        with st.expander('💡 Why is this score considered High Risk?'):
            st.write(f"""
                     In our preventative screening model, the high-risk threshold is strictly set at **{threshold_score:.1f}**. 
                     This is intentionally lower than 50 to cast a wider safety net (prioritizing Recall). 
                     A score of {risk_score:.1f} means your health indicators share significant patterns with diagnosed patients, warranting early preventative care.
                     """)
    else:
        st.success(f'✅ **Low Risk** (Risk Score: {risk_score:.1f} / 100)')
        st.progress(high_risk_prob) # Visual indicator
        st.write('Please maintain your current healthy lifestyle.')

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
        y=alt.Y('Importance', title='Global Impact Weight'),
        color=alt.Color('Importance', scale=alt.Scale(scheme='blues'), legend=None)
    )

    st.altair_chart(chart, use_container_width=True)

    st.info("""
            **Global Feature Importance**\n
            The chart above shows the general criteria that AI uses to evaluate health risks across all patients.
            Factors like GenHlth and HighBP are universally the strongest predictors in this model.
            """)
    
st.markdown('---')

# Use expander to hide technical metrics from general users but keep them accessible for reviewers
with st.expander('🔍 Model Performance Metrics (For Reviewers)'):
    st.write('This section provides quantitative evaluation metrics of the trained Random Forest model.')

    # Display metrics neatly in a row
    col1, col2, col3, col4 = st.columns(4)

    col1.metric('Accuracy', '70.0%')
    col2.metric('Recall', '79.0%')
    col3.metric('Precision', '34.0%')
    col4.metric('ROC-AUC', '0.811')

    st.info('The model was trained on the BRFSS dataset using a Random Forest algorithm with a comprehensive preprocessing pipeline.')