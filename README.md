# 🩺 Aplikasi Prediksi Risiko Diabetes — Deployment Model Machine Learning

Dokumentasi singkat untuk aspek penilaian **"Melakukan deployment model dalam bentuk
aplikasi web sederhana"** — berdasarkan Laporan Akhir Kelompok 8
*(Analisis Prediksi Penyakit Diabetes Menggunakan Pendekatan Machine Learning)*.

## 1. Ringkasan
Aplikasi web sederhana berbasis **Streamlit** yang men-deploy model **Random Forest**
(hasil hyperparameter tuning dengan `GridSearchCV`, sesuai Bab 4 & 11 laporan) untuk
memprediksi risiko diabetes pasien secara interaktif.

| Item | Keterangan |
|---|---|
| Framework deployment | Streamlit |
| Model | Random Forest Classifier (tuned) |
| Library model | Scikit-Learn, disimpan dengan Joblib |
| Dataset | Pima Indians Diabetes Dataset |
| Akurasi model (test set) | ± 0.73 |
| Fitur input | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age |
| Output | Prediksi kelas (Diabetes / Tidak Diabetes) + tingkat keyakinan (%) |

## 2. Struktur File
```
diabetes-app/
├── diabetes.csv           # dataset training
├── train_model.py         # script training & menyimpan model
├── app.py                 # aplikasi web Streamlit
├── requirements.txt       # daftar dependensi
├── model_diabetes.pkl     # model hasil training (dihasilkan otomatis)
├── scaler_diabetes.pkl    # scaler untuk preprocessing input (dihasilkan otomatis)
└── feature_columns.pkl    # urutan kolom fitur (dihasilkan otomatis)
```

## 3. Alur Sistem
1. Pengguna memasukkan data kesehatan pasien lewat form pada halaman web.
2. Aplikasi melakukan preprocessing (scaling) dengan `scaler_diabetes.pkl`, sama
   persis seperti proses saat training — agar konsisten.
3. Data yang sudah diproses dikirim ke `model_diabetes.pkl` untuk diprediksi.
4. Hasil prediksi (kelas + probabilitas) ditampilkan langsung di halaman web.

## 4. Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
python train_model.py      # melatih ulang & menyimpan model (sekali saja)
streamlit run app.py       # menjalankan aplikasi web
```
Aplikasi akan terbuka otomatis di `http://localhost:8501`.

## 5. Cara Mendapatkan Link Publik (Streamlit Community Cloud — gratis)
Karena deployment publik memerlukan akun GitHub & Streamlit milik Anda sendiri,
berikut langkah agar Anda memiliki **link aplikasi yang bisa dibagikan** (± 5 menit):

1. Buat repository baru di GitHub (misal: `diabetes-prediction-app`), lalu upload
   seluruh isi folder `diabetes-app/` (termasuk file `.pkl` hasil training, atau
   biarkan Streamlit Cloud menjalankan `train_model.py` sekali saat build).
2. Buka **[share.streamlit.io](https://share.streamlit.io)** → login dengan akun GitHub.
3. Klik **"New app"** → pilih repository yang baru dibuat → pilih branch (`main`) →
   isi **Main file path** dengan `app.py`.
4. Klik **Deploy**. Streamlit Cloud akan otomatis meng-install `requirements.txt`
   dan menjalankan aplikasi.
5. Setelah proses build selesai (± 1–2 menit), Anda akan mendapatkan link publik
   dengan format:
   ```
   https://<nama-app>-<username>.streamlit.app
   ```
6. Link inilah yang dicantumkan sebagai **"Link Aplikasi"** pada laporan/tugas.

> 💡 Alternatif lain: **Hugging Face Spaces** (pilih SDK "Streamlit") atau
> **Render.com** juga bisa digunakan dengan cara serupa jika ingin platform lain.

## 6. Catatan Etis (sesuai Bab 12 Laporan — Refleksi Etis)
- Aplikasi ini adalah **alat bantu skrining awal (decision support system)**,
  bukan pengganti diagnosis medis dari tenaga profesional.
- Data yang dimasukkan pengguna tidak disimpan oleh aplikasi (diproses secara
  langsung/*real-time* saja).
- Perlu evaluasi berkala terhadap performa & potensi bias model apabila
  digunakan pada populasi pasien yang berbeda dari data training.
