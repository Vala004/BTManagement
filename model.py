import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

# ---------------- Load dataset ----------------
df = pd.read_csv("train_cleaned_final.csv")
df = df.select_dtypes(include=[np.number])     # use only numeric features

y = df["stress_class"]
X = df.drop(columns=["stress_class"])

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Model function ----------------
def run_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')

    print(f"{name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return acc, model

# ---------------- Models ----------------
models = [
    ("LightGBM", LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=40)),
    ("CatBoost", CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, verbose=0)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1)),
    ("LogisticRegression", LogisticRegression(max_iter=300, n_jobs=-1)),
    ("LinearSVC", LinearSVC(C=1.0)),
    ("XGBoost", XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        num_class=5, objective="multi:softmax", eval_metric="mlogloss"
    ))
]

# ---------------- Train & save best ----------------
best_acc = -1
best_name = None
best_model = None

for name, model in models:
    acc, trained_model = run_model(name, model)
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = trained_model

joblib.dump(best_model, f"best_model_{best_name}.pkl")
print(f"\nBest model saved: best_model_{best_name}.pkl")
