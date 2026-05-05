import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class DiabetesPredictor:
    def __init__(self):
        """
        Initialize the class with empty model attributes.
        """
        self.model = None
        self.feature_names = None

    def load_model(self, model_path):
        """
        Load the pre-trained model and extract feature names.
        """
        self.model = joblib.load(model_path)
        try:
            self.feature_names = [
            'BMI',                               # Skewed feature
            'GenHlth', 'MentHlth', 'PhysHlth',   # Numerical/Ordinal features
            'Age', 'Education', 'Income',
            'HighBP', 'HighChol', 'CholCheck',   # Binary features
            'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
            'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 
            'NoDocbcCost', 'DiffWalk', 'Sex'
            ]
        except AttributeError:
            self.feature_names = None
        print(f'Successfully loaded model from: {model_path}')
    
    def predict(self, data):
        """
        Predict the target class based on input features.
        If data is already a DataFrame, use it directly.
        If it is a list, convert it to a DataFrame with feature names.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data], columns=self.feature_names)
        prediction = self.model.predict(data)
        
        return prediction[0]
    
    def get_feature_importance(self):
        """
        Extracts and returns feature importances from the trained pipeline 
        as a sorted pandas Series.
        """
        # Access the classifier from the last step of the pipeline
        importances = self.model.named_steps['classifier'].feature_importances_

        # Map importances to their corresponding feature names
        # Note: feature_names follows the concatenation order of the preprocessor 
        # (skewed -> num_ord -> binary)
        importance_series = pd.Series(importances, index=self.feature_names)

        return importance_series.sort_values(ascending=False)
    
    def predict_proba(self, data):
        """
        Predict the probabilities of the target classes based on input features.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data], columns=self.feature_names)

        probabilities = self.model.predict_proba(data)

        return probabilities[0]