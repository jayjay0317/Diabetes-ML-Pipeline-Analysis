import streamlit as st
import pandas as pd
from model_handler import DiabetesPredictor

# Page configuration
st.set_page_config(page_title='Diabetes Risk Predictor', layout='centered')

# Load model with caching to optimize performance
@st.cache_resource
def load_model():
    predictor = DiabetesPredictor()
    predictor.load_model('notebooks/diabetes_rf_model.pkl')
    return predictor