# Stress Level Classification Using ML

## 📌 Project Summary
Built a machine learning pipeline to classify human stress into 5 levels using HRV (Heart Rate Variability) data.

## 🚀 Key Features
- Complete end-to-end ML pipeline (data cleaning → preprocessing → model training)
- Handles skewness, correlation removal, scaling and feature selection
- Tested 6 ML models, best accuracy: **LightGBM — 99.3%**

## 📂 Files
- Script.ipynb — Data preprocessing and feature engineering
- model.py — Final training script for classification
- best_model_LightGBM.pkl — Saved trained model

## 🔧 Tech Stack
Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, CatBoost

## 📊 Performance
| Model | Accuracy |
|-------|----------|
| LightGBM | 0.993 |
| RandomForest | 0.984 |
| XGBoost | 0.959 |
| CatBoost | 0.900 |
| Logistic Regression | 0.683 |
| Linear SVC | 0.655 |

## 📎 Dataset
Original HRV dataset (not included due to size restrictions).

## 🧠 Future Scope
- Deploy stress prediction website
- Create live stress monitoring dashboard
- Replace engineered features with deep learning (LSTM / Transformers)

