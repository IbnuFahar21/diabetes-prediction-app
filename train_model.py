"""
train_model.py
Melatih model Random Forest untuk prediksi risiko diabetes,
lalu menyimpan model (.pkl) dan scaler (.pkl) agar bisa dipakai
oleh aplikasi web Streamlit (app.py).

Dataset : Pima Indians Diabetes Dataset (diabetes.csv)
Fitur   : Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age
Target  : Outcome (0 = Tidak Diabetes, 1 = Diabetes)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# 1. LOAD DATASET
df = pd.read_csv("diabetes.csv")

# 2. HANDLING MISSING VALUES (nilai 0 pada kolom berikut tidak valid secara medis)
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, pd.NA)
df[cols_with_invalid_zero] = df[cols_with_invalid_zero].fillna(df[cols_with_invalid_zero].median())

# 3. SPLIT FITUR & TARGET
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. NORMALISASI (disimpan agar preprocessing input user konsisten dengan training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. HYPERPARAMETER TUNING (Random Forest, sesuai laporan)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameter :", grid_search.best_params_)

# 6. EVALUASI MODEL FINAL
y_pred = best_rf.predict(X_test_scaled)

print("\n=== HASIL EVALUASI MODEL (Random Forest - Tuned) ===")
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 3))
print("Precision :", round(precision_score(y_test, y_pred), 3))
print("Recall    :", round(recall_score(y_test, y_pred), 3))
print("F1-Score  :", round(f1_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# 7. SIMPAN MODEL & SCALER UNTUK DEPLOYMENT
joblib.dump(best_rf, "model_diabetes.pkl")
joblib.dump(scaler, "scaler_diabetes.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel, scaler, dan daftar fitur berhasil disimpan:")
print("- model_diabetes.pkl")
print("- scaler_diabetes.pkl")
print("- feature_columns.pkl")
