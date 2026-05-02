from flask import Flask, request, jsonify
from model_handler import DiabetesPredictor

app = Flask(__name__)

# Initialize and load the predictor
MODEL_PATH = "../notebooks/diabetes_rf_model.pkl"
predictor = DiabetesPredictor()
predictor.load_model(MODEL_PATH)

@app.route('/')
def home():
    """
    Check if the server is running.
    """
    return "Diabetes Prediction Server is Running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive data and return prediction.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract features (Expecting a list of 21 values)
        features = data.get('features')

        # Get prediction result from our OOP handler
        result = predictor.predict(features)

        # Return the result as JSON
        return jsonify({
            'status': 'success',
            'prediction': int(result)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)