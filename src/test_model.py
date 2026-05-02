from model_handler import DiabetesPredictor

# Define the model path
# We use '../' to go up one level to the root, then into the notebooks folder
MODEL_PATH = '../notebooks/diabetes_rf_model.pkl'

def run_test():
    # Initialize the predictor
    predictor = DiabetesPredictor()

    # Load the model
    predictor.load_model(MODEL_PATH)
    
    # Create dummy patient data with 21 features (matching X_train)
    sample_data = [0.0] * 21

    # Perform prediction
    print('Testing prediction with sample data...')
    result = predictor.predict(sample_data)

    print(f'Prediction Result: {result}')

if __name__ == '__main__':
    run_test()