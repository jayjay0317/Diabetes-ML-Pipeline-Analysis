import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class DiabetesPredictor:
    def __init__(self, model_path):
        """
        Initialize the predictor by loading a pre-trained model.
        """
        self.model = joblib.load(model_path)
        # Get feature names from the model to create the DataFrame
        # with correct column names
        try:
            self.feature_names = self.model.feature_names_in_
        except AttributeError:
            self.feature_names = None

        print(f'Successfully loaded model from: {model_path}')
    
    def predict(self, features):
        """
        Predict the target class based on input features.
        Convert input list to DataFrame before performing prediction.
        """
        # Create a DataFrame because the model expects feature names
        data = pd.DataFrame([features], columns=self.feature_names)
        
        prediction = self.model.predict(data)
        return prediction[0]