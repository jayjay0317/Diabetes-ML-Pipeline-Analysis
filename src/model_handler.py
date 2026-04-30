import joblib
import numpy as np
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class DiabetesPredictor:
    def __init__(self, model_path):
        """
        Initialize the predictor by loading a pre-trained model.
        """
        self.model = joblib.load(model_path)
        print(f'Successfully loaded model from: {model_path}')
    
    def predict(self, features):
        """
        Predict the target class based on input features.
        """
        # Reshape to 2D array as required by Scikit-learn
        data = np.array(features).reshape(1, -1)
        prediction = self.model.predict(data)
        return prediction[0]