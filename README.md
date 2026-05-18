# 🏥 Diabetes Health Indicators Analysis & Deployment

An end-to-end data analysis and machine learning pipeline for diabetes risk prediction using the BRFSS 2015 dataset. This project addresses critical production and analytical challenges including severe class imbalance, non-linear clinical feature interactions, and environment disparity. These bottlenecks are resolved through robust statistical evaluation and a robust multi-container serving infrastructure deployed on AWS using Docker.

---

## 🌐 Deployment
Rather than relying on automated cloud tools, this system is fully deployed on a self-managed AWS EC2 instance running a clean Linux Ubuntu environment.  
🚀 **[Launch Live Demo via AWS](http://3.131.160.92:8501)**

---

## 🏗️ System Architecture

The production infrastructure separates the frontend and backend into isolated environments using a modular multi-container architecture.

* **Frontend Container** runs an interactive Streamlit web dashboard that captures user inputs and communicates with the inference server.
* **Backend Container** operates a Flask API engine to provide fast and lightweight model predictions.
* **Orchestration** utilizes Docker Compose to manage and bridge both containers within a single secure virtual network.

---

## 🎯 Key Technical Solutions

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

## ⚙️ Machine Learning Pipeline

### 1. Advanced Preprocessing with Scikit-learn Pipelines

I constructed a robust preprocessing pipeline to ensure data integrity and prevent data leakage:
- **Mathematical Safety with `log1p`**: Applied Log Transformation to highly skewed numerical features like **BMI** to normalize distribution and stabilize model training.
- **Standardization**: Integrated `StandardScaler` within the pipeline to ensure ordinal and continuous features are on a comparable scale.
- **Automated Column Transformation**: Utilized `ColumnTransformer` to apply specific transformations (Log, Scaling, or Passthrough) based on feature types, ensuring a seamless flow from raw data to inference.

### 2. Optimized Target Engineering
- **Target Binarization**: To improve clinical utility, I merged 'Prediabetes' and 'Diabetes' into a single 'At Risk' category, transforming a complex multi-class problem into a high-performing binary classification task.

---

## 📊 Model Performance

The final tuned Random Forest model demonstrates strong generalization and clinical utility.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.811** | Consistent discriminative power across unseen data. |
| **Recall** | **79.0%** | Effectively identifies the majority of at-risk individuals. |
| **Precision** | **34.0%** | Acceptable trade-off for early-stage preventative screening. |
| **F1-Score** | **0.470** | Balanced performance considering class imbalance. |
| **Accuracy** | **70.0%** | Overall correct predictions for the binarized classes. |

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
