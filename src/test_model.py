from model_handler import DiabetesPredictor

# Define the relative path to the trained model
MODEL_PATH = 'notebooks/diabetes_rf_model.pkl'

def run_model_test():
    """
    Unit test for the DiabetesPredictor class to ensure all methods
    (predict, predict_proba, get_feature_importance) function correctly.
    """
    print('--- Starting Unit Test for model_handler.py ---')

    # 1. Initialize and load the predictor
    predictor = DiabetesPredictor()
    predictor.load_model(MODEL_PATH)

    # 2. Create a realistic dummy patient data dictionary (21 features)
    sample_data = {
        'BMI': 35.5, 'GenHlth': 4.0, 'MentHlth': 15.0, 'PhysHlth': 20.0,
        'Age': 10.0, 'Education': 3.0, 'Income': 2.0, 'HighBP': 1.0,
        'HighChol': 1.0, 'CholCheck': 1.0, 'Smoker': 1.0, 'Stroke': 0.0,
        'HeartDiseaseorAttack': 1.0, 'PhysActivity': 0.0, 'Fruits': 0.0,
        'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0,
        'NoDocbcCost': 1.0, 'DiffWalk': 1.0, 'Sex': 1.0
    }

    # 3. Test exact class prediction
    print('\n[Test 1] Testing predict()...')
    pred_result = predictor.predict(sample_data)
    print(f'-> Predicted Class: {pred_result}')

    # 4. Test probability prediction
    print('\n[Test 2] Testing predict_proba()...')
    proba_result = predictor.predict_proba(sample_data)
    print(f'-> Prediction Probabilities: {proba_result}')

    # 5. Test feature importance extraction
    print('\n[Test 3] Testing get_feature_importance()...')
    importance_result = predictor.get_feature_importance()
    print(f'-> Top 5 Important Features:\n{importance_result.head(5)}')

    print('\n--- Unit Test Completed Successfully ---')

if __name__ == '__main__':
    run_model_test()







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