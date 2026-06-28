"""
Home.py — Halaman Utama Aplikasi Prediksi Harga Properti California
Pertemuan 14: Deploy ML App dengan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data untuk overview ──────────────────────────────────────────────────
@st.cache_data
def load_overview():
    """Load California Housing untuk statistik overview di halaman utama."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df

df = load_overview()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏠 California Housing Price Predictor")
st.markdown(
    """
    Selamat datang di aplikasi **Prediksi Harga Properti California** — 
    proyek Machine Learning end-to-end berbasis data sensus 1990.

    Aplikasi ini dibangun menggunakan **Streamlit** + **scikit-learn** sebagai
    bagian dari Pertemuan 14 mata kuliah Python Machine Learning.
    """
)

st.divider()

# ── Metrics overview ──────────────────────────────────────────────────────────
st.subheader("📊 Ringkasan Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Sampel",
        value=f"{len(df):,}",
        help="Jumlah blok sensus California"
    )
with col2:
    st.metric(
        label="Rata-rata Harga",
        value=f"${df['MedHouseVal'].mean() * 100_000:,.0f}",
        help="Median harga rumah dalam USD"
    )
with col3:
    st.metric(
        label="Harga Tertinggi",
        value=f"${df['MedHouseVal'].max() * 100_000:,.0f}",
        help="Harga median rumah tertinggi dalam dataset"
    )
with col4:
    st.metric(
        label="Jumlah Fitur",
        value="8",
        help="Fitur input untuk model prediksi"
    )

st.divider()

# ── Deskripsi dataset ─────────────────────────────────────────────────────────
st.subheader("📋 Tentang Dataset")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(
        """
        **California Housing Dataset** berisi data dari sensus California 1990.
        Setiap baris merepresentasikan satu **blok sensus** — area kecil
        berpenduduk 600–3.000 orang.

        Dataset ini sering digunakan untuk belajar regresi karena:
        - Ukurannya cukup besar (20.640 sampel)
        - Fiturnya bermakna secara nyata (pendapatan, usia rumah, lokasi)
        - Target yang jelas (harga rumah dalam $100,000)
        """
    )

with col_right:
    fitur_desc = pd.DataFrame({
        'Fitur': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                  'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'Deskripsi': [
            'Median pendapatan (×$10K)',
            'Median usia rumah (tahun)',
            'Rata-rata kamar per rumah tangga',
            'Rata-rata kamar tidur per rumah tangga',
            'Jumlah penduduk blok',
            'Rata-rata penghuni per rumah tangga',
            'Garis lintang blok',
            'Garis bujur blok',
        ]
    })
    st.dataframe(fitur_desc, use_container_width=True, hide_index=True)

st.divider()

# ── Navigasi ke halaman lain ──────────────────────────────────────────────────
st.subheader("🗺️ Navigasi Aplikasi")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown(
        """
        ### 📊 EDA
        Eksplorasi distribusi data, korelasi antar fitur,
        dan peta geografis harga rumah di California.
        """
    )
    st.page_link("pages/1_EDA.py", label="Buka halaman EDA →", icon="📊")

with nav_col2:
    st.markdown(
        """
        ### 🎯 Prediksi
        Masukkan nilai fitur rumah dan dapatkan prediksi
        harga dari model Random Forest secara real-time.
        """
    )
    st.page_link("pages/2_Prediksi.py", label="Buka halaman Prediksi →", icon="🎯")

with nav_col3:
    st.markdown(
        """
        ### 🤖 Tentang Model
        Lihat performa model (MAE, RMSE, R²),
        feature importance, dan detail training.
        """
    )
    st.page_link("pages/3_Tentang_Model.py", label="Buka halaman Model →", icon="🤖")

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "🎓 Dibuat untuk Pertemuan 14 — Python Machine Learning | "
    "Dataset: sklearn.datasets.fetch_california_housing | "
    "Model: RandomForestRegressor"
)

# Animasi sambutan
if 'welcomed' not in st.session_state:
    st.balloons()
    st.session_state['welcomed'] = True
