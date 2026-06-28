# ============================================================
# Demo 03 — Widget Input
# Jalankan: streamlit run streamlit_apps/p13_demo/03_widgets.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Widget Input", page_icon="🎛️", layout="wide")
st.title("🎛️ Widget Input — Demo Interaktif")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    tema = st.radio("Tampilan", ["Terang", "Gelap"], horizontal=True)
    st.divider()
    st.caption("Semua widget di bawah bisa kamu coba!")

st.divider()

# ── Row 1: Teks & Angka ──────────────────────────────────
st.header("1. Input Teks & Angka")
col1, col2, col3 = st.columns(3)

with col1:
    nama = st.text_input("Nama proyek", placeholder="misal: Prediksi Churn")
    st.write(f"Kamu ketik: `{nama}`")

with col2:
    n_data = st.number_input("Jumlah data", min_value=10, max_value=100000, value=1000, step=100)
    st.write(f"Jumlah: **{n_data:,}** sampel")

with col3:
    deskripsi = st.text_area("Deskripsi model", placeholder="Tulis deskripsi singkat...", height=100)

st.divider()

# ── Row 2: Slider ─────────────────────────────────────────
st.header("2. Slider")
col1, col2 = st.columns(2)

with col1:
    k = st.slider("Jumlah cluster K (K-Means)", min_value=2, max_value=10, value=3)
    st.info(f"K yang dipilih: **{k}** cluster")

with col2:
    rentang = st.slider("Rentang usia responden", min_value=17, max_value=65, value=(20, 35))
    st.info(f"Usia: **{rentang[0]}** – **{rentang[1]}** tahun")

st.divider()

# ── Row 3: Pilihan ────────────────────────────────────────
st.header("3. Pilihan (Selectbox, Multiselect, Radio)")
col1, col2, col3 = st.columns(3)

with col1:
    model = st.selectbox("Pilih Model ML", ["Random Forest", "SVM", "Logistic Regression", "XGBoost"])
    st.write(f"Model dipilih: **{model}**")

with col2:
    fitur = st.multiselect("Pilih Fitur", ["Usia", "Pendapatan", "Pendidikan", "Lokasi", "Pekerjaan"],
                           default=["Usia", "Pendapatan"])
    st.write(f"Fitur terpilih: {len(fitur)} fitur")

with col3:
    metrik = st.radio("Metrik Evaluasi", ["Accuracy", "F1-Score", "AUC-ROC"], index=1)
    st.write(f"Metrik: **{metrik}**")

st.divider()

# ── Row 4: Toggle, Button, File Upload ───────────────────
st.header("4. Toggle, Button & Upload")
col1, col2 = st.columns(2)

with col1:
    debug = st.toggle("Mode Debug")
    if debug:
        st.warning("⚠️ Mode debug aktif — semua output ditampilkan")

    normalize = st.checkbox("Normalisasi fitur sebelum training?", value=True)
    if normalize:
        st.info("StandardScaler akan digunakan")

with col2:
    uploaded = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ File dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")
        st.dataframe(df.head(3))
    else:
        st.info("Belum ada file yang diupload")

st.divider()

# ── Demo: Nilai semua widget ──────────────────────────────
with st.expander("🔍 Lihat nilai semua widget (untuk debugging)"):
    st.json({
        "nama": nama,
        "n_data": n_data,
        "k": k,
        "rentang_usia": list(rentang),
        "model": model,
        "fitur": fitur,
        "metrik": metrik,
        "debug": debug,
        "normalize": normalize,
    })
