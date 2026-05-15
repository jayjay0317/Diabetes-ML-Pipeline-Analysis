from flask import Flask, request, jsonify
from model_handler import DiabetesPredictor

app = Flask(__name__)

# Initialize and load the predictor
MODEL_PATH = "notebooks/diabetes_rf_model.pkl"
predictor = DiabetesPredictor()
predictor.load_model(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Diabetes Prediction API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract probability of the positive class (High Risk)
        probabilities = predictor.predict_proba(data)
        high_risk_prob = float(probabilities[1])

        return jsonify({
            'status': 'success',
            'high_risk_probability': high_risk_prob
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/importance', methods=['GET'])
def feature_importance():
    try:
        # Extract top 10 features for UI visualization
        importances = predictor.get_feature_importance().head(10)

        # Convert to List of Dictionaries to prevent alphabetical sorting
        importance_list = [
            {'Feature': feature, 'Importance': float(importance)}
            for feature, importance in importances.items()
        ]

        return jsonify({
            'status': 'success',
            'feature_importance': importance_list
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)