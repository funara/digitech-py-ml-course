# ============================================================
# Demo 04 — Layout (columns, sidebar, tabs, expander)
# Jalankan: streamlit run streamlit_apps/p13_demo/04_layout.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Layout Demo", page_icon="📐", layout="wide")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("📐 Layout Demo")
    st.divider()
    layout_demo = st.radio(
        "Pilih Demo Layout",
        ["Columns", "Tabs", "Expander", "Sidebar"]
    )
    st.divider()
    st.caption("Sidebar cocok untuk: filter, pengaturan, navigasi")

st.title("📐 Komponen Layout Streamlit")
st.divider()

# ── Demo Columns ───────────────────────────────────────────
st.header("1. st.columns() — Tata Letak Horizontal")

# Proporsi sama
st.subheader("Sama lebar (columns(3))")
col1, col2, col3 = st.columns(3)
col1.metric("MAE",  "12.34", "-1.2")
col2.metric("RMSE", "18.56", "-2.1")
col3.metric("R²",   "0.894", "+0.03")

st.subheader("Proporsi berbeda (columns([1, 2, 1]))")
kiri, tengah, kanan = st.columns([1, 2, 1])
with kiri:
    st.info("Kolom kiri (sempit) — cocok untuk label atau ikon")
with tengah:
    st.success("Kolom tengah (lebar) — cocok untuk chart atau tabel utama")
with kanan:
    st.warning("Kolom kanan (sempit)")

st.divider()

# ── Demo Tabs ─────────────────────────────────────────────
st.header("2. st.tabs() — Konten Berlapis")
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visualisasi", "⚙️ Model Info"])

with tab1:
    np.random.seed(42)
    df = pd.DataFrame({
        'Fitur A': np.random.randn(20),
        'Fitur B': np.random.randn(20),
        'Label':   np.random.choice(['Kelas 0', 'Kelas 1'], 20)
    })
    st.dataframe(df, use_container_width=True)

with tab2:
    st.write("Di sini bisa ditaruh chart, plot, atau visualisasi interaktif")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['Fitur A'], df['Fitur B'],
               c=['#3498db' if l == 'Kelas 0' else '#e74c3c' for l in df['Label']], alpha=0.7, s=60)
    ax.set_xlabel("Fitur A")
    ax.set_ylabel("Fitur B")
    ax.set_title("Scatter Plot Fitur A vs Fitur B")
    st.pyplot(fig)

with tab3:
    st.json({
        "model": "Random Forest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "train_accuracy": 0.962,
        "test_accuracy": 0.914
    })

st.divider()

# ── Demo Expander ─────────────────────────────────────────
st.header("3. st.expander() — Konten Dapat Disembunyikan")

with st.expander("📋 Klik untuk melihat detail preprocessing"):
    st.markdown("""
    **Langkah Preprocessing:**
    1. Drop kolom yang memiliki missing value > 30%
    2. Imputasi median untuk fitur numerik
    3. One-hot encoding untuk fitur kategorikal
    4. StandardScaler untuk normalisasi
    5. Train/test split 80:20
    """)

with st.expander("💡 Tips penggunaan model ini"):
    st.info("Model ini dilatih pada data 2020–2023. Performa mungkin turun untuk data di luar rentang tersebut.")
