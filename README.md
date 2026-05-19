# 🏥 Diabetes Health Indicators Analysis & Deployment

An end-to-end data analysis and machine learning pipeline for diabetes risk prediction using the BRFSS 2015 dataset. This project addresses critical production and analytical challenges including severe class imbalance, non-linear clinical feature interactions, and environment disparity. These bottlenecks are resolved through robust statistical evaluation and a robust multi-container serving infrastructure deployed on AWS using Docker.

---

## 🌐 Deployment
Rather than relying on automated cloud tools, this system is fully deployed on a self-managed AWS EC2 instance running a clean Linux Ubuntu environment.  
🚀 **[Launch Live Demo via AWS](http://3.131.160.92:8501)**

---

## 🏗️ System Architecture

The production infrastructure separates the frontend and backend into isolated environments using a modular multi-container architecture.

* **Frontend Container**: runs an interactive Streamlit web dashboard that captures user inputs and communicates with the inference server.
* **Backend Container**: operates a Flask API engine to provide fast and lightweight model predictions.
* **Orchestration**: utilizes Docker Compose to manage and bridge both containers within a single secure virtual network.

---

## 🎯 Key Technical Solutions

### 1. Advanced Imbalance Handling & Strategic Pivot
- **The Accuracy Paradox**: Initial baseline models achieved **83% accuracy** but failed clinically with **0% recall** for the prediabetes class due to severe data skewness.
- **Target Binarization**: Strategically merged 'Prediabetes' and 'Diabetes' into a single 'At Risk' class to prioritize preventative screening efficacy and improve overall model stability.
- **Cost-Sensitive Learning**: Implemented balanced class weights (`class_weight='balanced'`) in Random Forest to aggressively penalize misclassifications of minority-class patients.

### 2. Clinical Threshold Optimization
- **Recall-First Strategy**: Prioritized **Recall (79%)** over Precision (34%) to minimize False Negatives in a clinical screening context where missing a patient carries high risk.
- **Youden’s J Statistic**: Derived a mathematically optimal threshold of **0.419** from cross-validated ROC curves, replacing the default 0.5 to maximize sensitivity.

### 3. Production-Ready Software Engineering
- **OOP-Based Refactoring**: Encapsulated the prediction engine into a modular `DiabetesPredictor` class for improved maintainability and scalable deployment.
- **Real-Time Feature Engineering**: Developed dynamic mapping functions to translate raw user inputs (e.g., Age, Height, Weight) into BRFSS-standard categorical data on the fly.

---

## 🛠️ Technical Stack

- **Infrastructure & Deployment**: AWS EC2, Docker, Docker Compose, Ubuntu
- **Machine Learning & Data Processing**: Scikit-learn, Pandas, NumPy
- **API & Frontend**: Flask, Streamlit
- **Visualization**: Altair
- **Model Persistence**: Joblib
- **Version Control**: Git

---

## ⚙️ Machine Learning Pipeline

### Advanced Preprocessing with Scikit-learn Pipelines

I constructed a robust preprocessing pipeline to prevent data leakage during training and ensure strict consistency in data transformations during live inference environments.
- **Prevention of Data Leakage**: Encapsulated all preprocessing steps inside a Scikit-learn Pipeline object to ensure that scaling parameters are computed strictly on training folds, preventing information from validation folds from leaking into the training process.
- **Mathematical Safety with `log1p`**: Applied Log Transformation to highly skewed numerical features like BMI to normalize distribution and stabilize model training.
- **Standardization**: Integrated `StandardScaler` within the pipeline to ensure ordinal and continuous features are on a comparable scale.
- **Automated Column Transformation**: Utilized `ColumnTransformer` to apply specific transformations based on feature types, ensuring a seamless flow from raw data to production inference.

---

## 📊 Model Performance

The final tuned Random Forest model demonstrates strong generalization and clinical utility, aligning with preventative healthcare screening standards.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.811** | Consistent discriminative power across unseen data |
| **Recall** | **79.0%** | Effectively identifies the majority of at-risk individuals to minimize false negatives |
| **Precision** | **34.0%** | Acceptable trade-off for early-stage preventative screening thresholds |
| **F1-Score** | **0.470** | Reflects the trade-off of maximizing Recall under severe class imbalance |
| **Accuracy** | **70.0%** | Overall correct predictions for the binarized target classes |

---

## 🖥️ Implementation Highlights

### **Explainable AI (XAI)**
The dashboard provides Global Feature Importance visualizations using Altair to help users understand that general health perception (GenHlth) and high blood pressure (HighBP) act as the primary risk drivers inside the random forest model.

### **Human-Centric UX**
- **Automatic BMI Calculation**: Simplifies the user experience by calculating BMI from user-provided height and weight.
- **Risk Score System**: Translates raw machine learning probabilities into an intuitive 100 point risk score system to communicate health risks clearly to non technical end users.

---

## 🧬 Project Structure

* `docker-compose.yml` defines the multi-container orchestration for frontend and backend services.
* `Dockerfile.flask` configures the environment for the Flask API inference server.
* `Dockerfile.streamlit` configures the environment for the interactive web dashboard application.
* `requirements.txt` lists the essential Python library dependencies.
* `notebooks/` contains exploratory data analysis, data cleaning, model optimization workflows, and the serialized `diabetes_rf_model.pkl` file.
* `src/model_handler.py` holds the core `DiabetesPredictor` class with feature importance logic.
* `src/app.py` runs the Flask API backend server providing the prediction endpoint.
* `src/streamlit_app.py` runs the interactive Streamlit user interface for real-time risk assessment.

---

## 🧪 How to Run

This project uses Docker Compose to manage the multi-container environment (recommended for consistent deployment). Follow these steps to set up and run the service locally.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jayjay0317/Diabetes-ML-Pipeline-Analysis.git
   # Navigate into the project directory
   cd Diabetes-ML-Pipeline-Analysis
   ```

2. **Launch the services**:
Execute the following command to build the containers and start the Flask API and Streamlit dashboard services simultaneously.
    ```bash
    docker compose up --build
    ```

3. **Access the application**:
Once the containers are running, you can access the services via your web browser:
* Streamlit Dashboard: http://localhost:8501
* Flask API: http://localhost:5000

4. **Stop the services**:
To shut down the containers, press Ctrl+C in your terminal or run:
    ```bash
    docker compose down
    ```




This project uses Docker Compose to manage the multi-container environment. Follow these steps to set up and run the service locally.

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

### ⚗️ How to Test API

Once the Flask server is running (`python src/app.py`), you can test the endpoint using the following commands:

1. **Windows (PowerShell)**:
```powershell
# Send sample request to local API
$body = @{ features = @(0,0,0,30,0,0,0,1,1,1,0,1,0,3,0,0,0,1,8,5,8) } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Body $body -ContentType "application/json"
```

2. **Mac / Linux (Terminal)**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,30,0,0,0,1,1,1,0,1,0,3,0,0,0,1,8,5,8]}'
```

3. Expected Response:

A successful request returns a JSON object with the prediction result:

```json
{
  "prediction": 0,
  "status": "success"
}
```
Where:
- `0` = Normal
- `1` = At Risk
