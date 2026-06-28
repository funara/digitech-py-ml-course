# Digitech Python Machine Learning Course

Repositori ini berisi materi pembelajaran, slide teori, latihan coding (Jupyter Notebook), dan aplikasi web interaktif (Streamlit) untuk mata kuliah **Python Programming (Machine Learning)** di Universitas Teknologi Digital.

## 📁 Struktur Repositori

```
├── README.md               <- Panduan utama repositori ini
├── requirements.txt        <- Dependensi library Python untuk proyek
├── .gitignore              <- File konfigurasi untuk mengecualikan file lokal dari Git
│
├── courses/                <- Folder utama seluruh materi pembelajaran
│   ├── notebooks/          <- Jupyter Notebook untuk praktek coding
│   │   ├── python_crashcourse/  <- Pengayaan dasar pemrograman Python (pertemuan 00-03)
│   │   ├── 06_python_ml_data_cleaning.ipynb
│   │   ├── 07_python_ml_data_preparation.ipynb
│   │   ├── 09_python_ml_classification.ipynb
│   │   ├── 10_python_ml_regression.ipynb
│   │   ├── 11_python_ml_clustering.ipynb
│   │   ├── 12_python_ml_timeseries.ipynb
│   │   ├── 13_python_ml_streamlit_basics.ipynb
│   │   └── 14_python_ml_streamlit_deploy.ipynb
│   │
│   ├── slides/             <- Slide presentasi materi teori & Visual ML (Orange)
│   │   ├── 01_python_ml_intro1.pdf
│   │   ├── 02_python_ml_intro2.pdf
│   │   ├── 03_orange_ml_regression.pdf
│   │   ├── 04_orange_ml_clustering.pdf
│   │   └── 05_orange_ml_timeseries.pdf
│   │
│   └── streamlit_apps/     <- Aplikasi web interaktif berbasis Streamlit
│       ├── p13_demo/       <- Demo dasar widgets dan kalkulator sederhana
│       └── p14_ml_app/     <- Aplikasi Machine Learning multipage (Home, EDA, Prediksi)
│
├── data/                   <- Tempat penyimpanan dataset
├── docs/                   <- Buku panduan, modul, dan handbook referensi belajar
├── models/                 <- Tempat penyimpanan model latih (.pkl, .joblib, dll.)
└── reports/                <- Laporan analisis dan visualisasi hasil evaluasi model
```

## 🗓️ Keselarasan Materi (RPS)

Seluruh materi di dalam folder `courses/` diselaraskan dengan Rencana Pembelajaran Semester (RPS):
1. **Minggu 1 - 5 (Teori & Visual ML)**: Mempelajari konsep dasar AI/ML serta eksplorasi visual algoritma klasifikasi, regresi, clustering, dan time series menggunakan aplikasi Orange (Slide teori berada di `courses/slides/`).
2. **Minggu 6 - 7 (Data Preparation)**: Praktek pemrograman Python untuk pembersihan data (`06_python_ml_data_cleaning.ipynb`) dan penyiapan data (`07_python_ml_data_preparation.ipynb`).
3. **Minggu 8**: Evaluasi Tengah Semester (ETS) - Pitching Proposal.
4. **Minggu 9 - 12 (Python ML)**: Implementasi algoritma ML menggunakan library `scikit-learn` pada notebook pertemuan 09 s.d 12.
5. **Minggu 13 - 14 (Streamlit)**: Pembuatan dashboard interaktif dan deployment aplikasi web ML ke Streamlit Cloud (`courses/streamlit_apps/`).
6. **Minggu 15 - 16**: Finalisasi proyek tim & Evaluasi Akhir Semester (App Fair).

---
*Selamat belajar dan berkesperimen dengan Machine Learning!* 🚀
