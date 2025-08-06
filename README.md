# 📉 Customer Churn Prediction (Telco Dataset)

Predict which customers are likely to leave a telecom service using advanced ML models. This end-to-end project involves data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning — achieving nearly perfect performance using **LightGBM**.

---

## 🚀 Project Overview

* **Goal**: Predict customer churn to help businesses take proactive steps for customer retention.
* **Dataset**: Fictional telecom dataset with 7,043 entries and 50+ features.
* **Final Model**: Tuned `LGBMClassifier` with outstanding performance (AUC = **0.999**).
* **Tech Stack**: Python, scikit-learn, LightGBM, XGBoost, seaborn, pandas, matplotlib.

---

## 📦 Dataset Details

* **Demographics**: Age, Gender, Location, Dependents.
* **Services**: Internet type, contract, tech support, streaming.
* **Usage & Charges**: Monthly Charges, Total Charges, Data Download.
* **Churn Label**: Binary target indicating churn (`Yes` / `No`).

---

## 🧠 Key Steps

### ✅ 1. Exploratory Data Analysis

* Missing values visualized via heatmap.
* Target variable distribution showed **26.5% churn rate**.
* Categorical and numerical features analyzed in detail.

### 🧹 2. Data Cleaning & Preprocessing

* Imputed missing values using business logic.
* Treated outliers using IQR capping.
* Removed irrelevant columns (e.g., `Customer ID`, `Churn Reason`).
* Handled rare categories and duplicates.

### 🏗️ 3. Feature Engineering

Created new features:

* `Total Services` subscribed
* `Service Level` (Low, Medium, High)
* `Customer Cluster` via K-Means
* `Has Tech Support or Protection`
* `Is Senior and On Contract`

### 🔁 4. Feature Encoding & Scaling

* One-hot encoded all categorical features (low cardinality).
* Standardized numerical features using `StandardScaler`.

### 🤖 5. Model Training & Evaluation

Trained and compared the following models:

| Model                 | Accuracy | F1 Score | ROC AUC |
| --------------------- | -------- | -------- | ------- |
| LGBMClassifier (Best) | 0.9844   | 0.9704   | 0.9990  |
| XGBoost Classifier    | 0.9844   | 0.9705   | 0.9989  |
| Logistic Regression   | 0.9844   | 0.9704   | 0.9988  |
| SVM                   | 0.9823   | 0.9660   | 0.9983  |
| Random Forest         | 0.9865   | 0.9743   | 0.9974  |

> ✅ **LGBMClassifier** was chosen as the final model due to its highest ROC-AUC.

### 🛠️ 6. Hyperparameter Tuning

Used `RandomizedSearchCV` with 5-fold cross-validation to tune:

* Learning rate
* Max depth
* Num leaves
* Regularization parameters

---

## 🎯 Key Insights

* 📉 Customers with **month-to-month contracts** and **no tech support** are more likely to churn.
* 🔌 `Service Level`, `Tenure`, and `Churn Score` are strong predictors.
* 📊 K-Means clustering helped uncover customer segments.

---

## 📁 Repository Structure

```
📦 customer-churn-prediction/
🗄️ data/
📂   └— telco.csv
🗄️ models/
📂   ├— best_churn_model.pkl
📂   └— lgbm_test_results.csv
🗄️ notebooks/
📂   └— churn_prediction_pipeline.ipynb
🗄️ visuals/
📂   ├— model_performance A.png
📂   └— model_performance B.png
🗄️ README.md
🗄️ requirements.txt
```

---

## 📄 Deliverables

* ✅ Trained model: `best_churn_model.pkl`
* ✅ Predictions: `lgbm_test_results.csv`
* ✅ Visuals: Confusion matrix, ROC curve, model comparison charts

---

## 🌐 Gradio App (Live Demo)

👉 Try the live app here:
**[Customer Churn Predictor – Hugging Face Space](https://huggingface.co/spaces/Hohenhiem/customer_churn_predictor)**

You can input the top 5 features and get real-time churn predictions powered by the trained LGBM model.

---

## 💡 Conclusion

This project demonstrates how thoughtful data cleaning, feature engineering, and model tuning can produce a high-performing churn prediction model. The insights are actionable and can help reduce churn by targeting high-risk customers with personalized strategies.

---
