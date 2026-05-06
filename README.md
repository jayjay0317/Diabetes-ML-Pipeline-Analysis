# 🏥 Diabetes Health Indicators Analysis & Deployment
An end-to-end machine learning pipeline for diabetes risk prediction using the BRFSS 2015 dataset. This project addresses critical real-world challenges: severe class imbalance, non-linear clinical feature interactions, and UX-driven model serving.

- log1p for mathematical safety
- class_weight='balanced'

🏥 Diabetes Health Indicators Analysis & Deployment
A project focused on building a machine learning model to predict diabetes risk and deploying it as a functional web service.

🛠️ Current Progress
Data Analysis & Modeling: Developed a binary classification model using Random Forest based on health indicator datasets.

OOP-based Refactoring: Transitioned procedural analysis code into a modular DiabetesPredictor class within model_handler.py for improved maintainability.

API Development: Built a web server using the Flask framework to facilitate communication between the model and external requests.

Model Serving: Successfully implemented a real-time inference system on a local environment (127.0.0.1:5000).

🚀 Key Features
Inference Endpoint: Provides a /predict route that accepts 21 health-related features in JSON format and returns immediate predictions.

Modular Architecture: Separated core model logic from the server interface to ensure system scalability.

Clean Code Practices: Documented the codebase with concise, meaningful English comments suitable for a professional portfolio.

🧪 How to Run
Start Server: Run python app.py within the src directory.

Test API: Send a POST request with feature data using PowerShell (Invoke-RestMethod) or CMD (curl).