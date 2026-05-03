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
            self.feature_names = self.model.feature_names_in_
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