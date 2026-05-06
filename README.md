# 🏥 Diabetes Health Indicators Analysis & Deployment

> **Live Demo:** [🚀 View my Streamlit App](https://jaewoo-diabetes-prediction.streamlit.app/)

An end-to-end machine learning pipeline for diabetes risk prediction using the BRFSS 2015 dataset. This project addresses critical real-world challenges: **severe class imbalance**, **non-linear clinical feature interactions**, and **UX-driven model serving**.

---

## 🚀 Key Technical Solutions

### 1. Advanced Imbalance Handling & Strategic Pivot
- **The Accuracy Paradox**: Initial baseline models achieved **83% accuracy** but failed clinically with **0% recall** for the prediabetes class due to severe data skewness.
- **Target Binarization**: Strategically merged 'Prediabetes' and 'Diabetes' into a single 'At Risk' class to prioritize preventative screening efficacy and improve model stability.
- **Cost-Sensitive Learning**: Implemented `class_weight='balanced'` in Random Forest to aggressively penalize misclassifications of minority-class patients.

### 2. Clinical Threshold Optimization
- **Recall-First Strategy**: Prioritized **Recall (79%)** over Precision (34%) to minimize False Negatives in a clinical screening context where missing a patient is high-risk.
- **Youden’s J Statistic**: Derived a mathematically optimal threshold of **0.419** from cross-validated ROC curves, replacing the default 0.5 to maximize sensitivity.

### 3. Production-Ready Software Engineering
- **OOP-Based Refactoring**: Encapsulated the prediction engine into a modular `DiabetesPredictor` class for improved maintainability and scalable deployment.
- **Real-Time Feature Engineering**: Developed dynamic mapping functions to translate raw user inputs (e.g., Age, Height, Weight) into BRFSS-standard categorical data.

---

## 🛠️ Technical Stack

- **Machine Learning**: Scikit-learn (Random Forest, Logistic Regression, Pipelines).
- **Data Engineering**: Pandas, NumPy (Log-transformation, Stratified Splitting).
- **Serving & UI**: Streamlit (Web Dashboard), Flask (REST API), Altair (Visualization).
- **Tools**: Joblib (Model Serialization), Git, VS Code, Anaconda.

---

## 📊 Model Performance

The final tuned Random Forest model demonstrates strong generalization and clinical utility.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.811** | Consistent discriminative power across unseen data. |
| **Recall** | **79.0%** | Effectively identifies the majority of at-risk individuals. |
| **Precision** | **34.0%** | Acceptable trade-off for early-stage preventative screening. |

---

## 🧬 Project Structure

- `notebooks/`: Contains EDA, data cleaning, and model optimization workflows.
- `src/model_handler.py`: Core `DiabetesPredictor` class with feature importance logic.
- `src/app.py`: Flask-based backend server providing the `/predict` REST API endpoint.
- `src/streamlit_app.py`: Interactive web UI for real-time risk assessment and reviewer metrics.

---

## 🖥️ Implementation Highlights

### **Explainable AI (XAI)**
The dashboard provides **Global Feature Importance** visualizations using Altair, helping users understand that `GenHlth` and `HighBP` are the primary risk drivers in the model.

### **Human-Centric UX**
- **Automatic BMI Calculation**: Simplifies the process by calculating BMI from user-provided height and weight.
- **Risk Score System**: Translates raw ML probabilities into a **100-point Risk Score** for more intuitive communication of health risks.

---

## 🧪 How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jayjay0317/Diabetes-ML-Pipeline-Analysis.git
   # Navigate into the project directory
   cd Diabetes-ML-Pipeline-Analysis
   ```

2. **Install Dependencies**:
    ```bash
    # Install required libraries for the project
    pip install -r requirements.txt
    ```

3. **Launch the Streamlit App**:
    ```bash
    # Start the interactive web interface
    streamlit run src/streamlit_app.py
    ```
4. **Test the Flask API**:
    ```bash
    # Start the Flask backend server for REST API testing
    python src/app.py
    ```


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