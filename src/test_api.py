import requests
import json

# Define the Flask API base URL
BASE_URL = 'http://localhost:5000'

def run_api_test():
    """
    Integration test to verify Flask API endpoints respond correctly 
    to HTTP requests.
    """
    print(f'--- Starting API Integration Test targeting {BASE_URL} ---')

    # Dummy payload for POST request
    payload = {
        'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 35.5,
        'Smoker': 1.0, 'Stroke': 0.0, 'HeartDiseaseorAttack': 0.0,
        'PhysActivity': 0.0, 'Fruits': 0.0, 'Veggies': 1.0,
        'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0,
        'GenHlth': 4.0, 'MentHlth': 15.0, 'PhysHlth': 20.0,
        'DiffWalk': 1.0, 'Sex': 1.0, 'Age': 10.0, 'Education': 3.0, 'Income': 2.0
    }

    # 1. Test the /predict POST endpoint
    print('\n[Test 1] Testing POST /predict endpoint...')
    try:
        pred_response = requests.post(f'{BASE_URL}/predict', json=payload)
        print(f'-> Status Code: {pred_response.status_code}')
        if pred_response.status_code == 200:
            print(f"-> High Risk Probability: {pred_response.json().get('high_risk_probability'):.4f}")
    except requests.exceptions.ConnectionError:
        print('-> Error: Could not connect. Is the Flask server running?')

    # 2. Test the /importance GET endpoint
    print('\n[Test 2] Testing GET /importance endpoint...')
    try:
        imp_response = requests.get(f'{BASE_URL}/importance')
        print(f'-> Status Code: {imp_response.status_code}')
        if imp_response.status_code == 200:
            features_list = imp_response.json().get('feature_importance', [])
            top_features = [item['Feature'] for item in features_list[:3]]

            print(f'-> Successfully retrieved importance. Top 3: {top_features}')
    except requests.exceptions.ConnectionError:
        print('-> Error: Could not connect. Is the Flask server running?')

    print('\n--- API Test Completed ---')

if __name__ == '__main__':
    run_api_test()