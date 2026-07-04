"""
app.py
Aplikasi Web (Streamlit) untuk deployment model Machine Learning
prediksi risiko diabetes.

Menjalankan lokal:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------------------------------
# LOAD MODEL, SCALER, DAN DAFTAR FITUR (di-cache agar tidak reload berulang)
# ----------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_diabetes.pkl")
    scaler = joblib.load("scaler_diabetes.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("🩺 Aplikasi Prediksi Risiko Diabetes")
st.markdown(
    """
    Aplikasi ini merupakan hasil **deployment model Machine Learning (Random Forest)**
    yang dikembangkan pada laporan *Analisis Prediksi Penyakit Diabetes Menggunakan
    Pendekatan Machine Learning* (Kelompok 8).

    Masukkan data kesehatan pasien di bawah ini, lalu klik **Prediksi** untuk
    mengetahui estimasi risiko diabetes.
    """
)

st.divider()

# ----------------------------------------------------
# FORM INPUT DATA PASIEN
# ----------------------------------------------------
st.subheader("📋 Data Kesehatan Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Kadar Insulin (Insulin)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=30, step=1)

st.divider()

# ----------------------------------------------------
# PREDIKSI
# ----------------------------------------------------
if st.button("🔍 Prediksi", use_container_width=True, type="primary"):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=feature_columns
    )

    # Preprocessing input sama seperti saat training (scaling)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Pasien **BERISIKO DIABETES**  \nTingkat keyakinan model: **{proba[1]*100:.1f}%**")
    else:
        st.success(f"✅ Pasien **TIDAK BERISIKO DIABETES**  \nTingkat keyakinan model: **{proba[0]*100:.1f}%**")

    with st.expander("Lihat detail probabilitas"):
        st.write(f"- Probabilitas Tidak Diabetes: {proba[0]*100:.2f}%")
        st.write(f"- Probabilitas Diabetes: {proba[1]*100:.2f}%")

    st.caption(
        "⚠️ Catatan: Hasil prediksi ini bersifat estimasi berdasarkan model Machine Learning "
        "dan hanya digunakan sebagai alat bantu skrining awal (decision support), "
        "bukan pengganti diagnosis medis oleh tenaga profesional."
    )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.divider()
st.caption("Model: Random Forest (tuned dengan GridSearchCV) · Dataset: Pima Indians Diabetes Dataset")
