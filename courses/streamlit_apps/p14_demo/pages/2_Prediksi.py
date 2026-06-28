"""
2_Prediksi.py — Halaman Prediksi Harga Rumah
Pertemuan 14: Deploy ML App dengan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi — California Housing",
    page_icon="🎯",
    layout="wide",
)

# ── Load data & train model (di-cache) ────────────────────────────────────────
@st.cache_data
def load_data():
    """Load California Housing sebagai DataFrame — di-cache."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df, housing.feature_names

@st.cache_resource
def train_model():
    """
    Latih Random Forest Regressor pada California Housing.
    Di-cache dengan @st.cache_resource karena model sklearn tidak serializable.
    Hanya dijalankan SEKALI — semua user berbagi model yang sama.
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training model Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1   # pakai semua CPU core
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Load data dan model
df, feature_names = load_data()
model, X_train, X_test, y_train, y_test = train_model()

# Rata-rata dataset untuk perbandingan
df_mean = df[list(feature_names)].mean()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 Prediksi Harga Rumah")
st.markdown(
    "Masukkan karakteristik blok sensus di sidebar, lalu klik **Prediksi** "
    "untuk mendapatkan estimasi harga median rumah dari model Random Forest."
)

# ── Sidebar: Input fitur ──────────────────────────────────────────────────────
st.sidebar.header("🏠 Input Fitur Rumah")
st.sidebar.markdown("Sesuaikan nilai fitur blok sensus:")

med_inc = st.sidebar.slider(
    "MedInc — Median Pendapatan (×$10K)",
    min_value=0.5, max_value=15.0,
    value=float(df_mean['MedInc']),
    step=0.1,
    help="Median pendapatan rumah tangga dalam blok sensus (dalam kelipatan $10,000)"
)

house_age = st.sidebar.slider(
    "HouseAge — Median Usia Rumah (tahun)",
    min_value=1.0, max_value=52.0,
    value=float(df_mean['HouseAge']),
    step=1.0,
    help="Median usia rumah dalam blok sensus"
)

ave_rooms = st.sidebar.slider(
    "AveRooms — Rata-rata Jumlah Kamar",
    min_value=1.0, max_value=10.0,
    value=float(min(df_mean['AveRooms'], 10.0)),
    step=0.1,
    help="Rata-rata jumlah kamar per rumah tangga"
)

ave_bedrms = st.sidebar.slider(
    "AveBedrms — Rata-rata Kamar Tidur",
    min_value=1.0, max_value=5.0,
    value=float(min(df_mean['AveBedrms'], 5.0)),
    step=0.1,
    help="Rata-rata jumlah kamar tidur per rumah tangga"
)

population = st.sidebar.number_input(
    "Population — Jumlah Penduduk Blok",
    min_value=3, max_value=35682,
    value=int(df_mean['Population']),
    step=100,
    help="Jumlah total penduduk dalam blok sensus"
)

ave_occup = st.sidebar.slider(
    "AveOccup — Rata-rata Penghuni per Rumah",
    min_value=1.0, max_value=10.0,
    value=float(min(df_mean['AveOccup'], 10.0)),
    step=0.1,
    help="Rata-rata jumlah penghuni per rumah tangga"
)

latitude = st.sidebar.slider(
    "Latitude — Garis Lintang",
    min_value=32.5, max_value=42.0,
    value=float(df_mean['Latitude']),
    step=0.1,
    help="Garis lintang blok sensus (32 = selatan, 42 = utara California)"
)

longitude = st.sidebar.slider(
    "Longitude — Garis Bujur",
    min_value=-124.0, max_value=-114.0,
    value=float(df_mean['Longitude']),
    step=0.1,
    help="Garis bujur blok sensus (-124 = barat pesisir, -114 = timur)"
)

# Susun input sebagai array untuk prediksi
input_features = np.array([[
    med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude
]])

# ── Tombol prediksi ──────────────────────────────────────────────────────────
st.subheader("Hasil Prediksi")

col_pred, col_info = st.columns([1, 2])

with col_pred:
    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        # Jalankan prediksi
        predicted_val = model.predict(input_features)[0]
        predicted_usd = predicted_val * 100_000

        # Simpan ke session_state
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []

        st.session_state['prediction_history'].append({
            'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude,
            'Prediksi (×$100K)': round(predicted_val, 3),
            'Prediksi (USD)': f"${predicted_usd:,.0f}"
        })

        st.session_state['last_prediction'] = predicted_val

    # Tampilkan hasil jika ada
    if 'last_prediction' in st.session_state:
        pred_val = st.session_state['last_prediction']
        pred_usd = pred_val * 100_000

        st.metric(
            label="Harga Prediksi",
            value=f"${pred_usd:,.0f}",
            help=f"Nilai mentah: {pred_val:.4f} × $100,000"
        )

        # Pesan berdasarkan nilai prediksi
        if pred_usd > 350_000:
            st.success("🏖️ Properti premium — di atas rata-rata California!")
        elif pred_usd > 200_000:
            st.info("🏡 Properti sedang — harga tipikal California.")
        else:
            st.warning("🏘️ Properti terjangkau — di bawah rata-rata California.")

with col_info:
    st.markdown("**Perbandingan Input vs Rata-rata Dataset:**")

    # Tabel perbandingan
    comparison_data = {
        'Fitur': list(feature_names),
        'Nilai Input': [
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ],
        'Rata-rata Dataset': [round(df_mean[f], 3) for f in feature_names],
    }
    df_comp = pd.DataFrame(comparison_data)
    df_comp['Selisih'] = (df_comp['Nilai Input'] - df_comp['Rata-rata Dataset']).round(3)
    df_comp['Status'] = df_comp['Selisih'].apply(
        lambda x: '▲ Di atas rata-rata' if x > 0 else ('▼ Di bawah rata-rata' if x < 0 else '= Rata-rata')
    )
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.divider()

# ── History prediksi ───────────────────────────────────────────────────────────
st.subheader("📋 Riwayat Prediksi")

if 'prediction_history' not in st.session_state or not st.session_state['prediction_history']:
    st.info("Belum ada prediksi. Klik tombol **Prediksi Sekarang** untuk memulai.")
else:
    df_history = pd.DataFrame(st.session_state['prediction_history'])
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    st.caption(f"Total prediksi: {len(df_history)} kali")

    col_clear, col_dl = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Hapus History", use_container_width=True):
            st.session_state['prediction_history'] = []
            if 'last_prediction' in st.session_state:
                del st.session_state['last_prediction']
            st.rerun()
    with col_dl:
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History sebagai CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
