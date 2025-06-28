# 📉 Customer Churn Prediction

This project builds a machine learning model to predict telecom customer churn using a dataset of 7,043 customers with 47 features.

---

## 🧠 Highlights

- End-to-end ML pipeline (EDA → Feature Engineering → Modeling → Tuning)
- Tuned Random Forest with 98.4% accuracy & 0.999 ROC-AUC
- Business insights included for churn reduction strategies
- Radar chart comparison of 4 models
- Saved final model and test results for deployment/future use

---

## 📊 Dataset

- Demographics, services, usage, contracts, and churn status.

---

## 🚀 Tech Stack

- Python, Pandas, NumPy, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Kaggle Notebook, Matplotlib, Joblib

---

## 📈 Model Results

| Model              | Accuracy | ROC-AUC |
|-------------------|----------|---------|
| Logistic Regression | 98.1%   | 0.9987  |
| Random Forest       | 98.2%   | 0.9989  |
| LightGBM            | 98.3%   | 0.9990  |
| **Tuned RF (Final)**| **98.4%** | **0.9990** |

---

## 📂 Files

- `churn_model_notebook.ipynb` – Full code and markdown explanation
- `tuned_random_forest_model.pkl` – Final trained model
- `rf_test_results.csv` – Exported predictions from best model
- `images/` – Contains charts like radar model comparison

---

## 📦 How to Run

```bash
git clone https://github.com/Saadmirza99114/customer-churn-prediction.git
cd customer-churn-prediction

pip install -r requirements.txt
jupyter notebook churn_model_notebook.ipynb
