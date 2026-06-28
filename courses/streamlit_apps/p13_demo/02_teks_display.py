# ============================================================
# Demo 02 — Komponen Teks & Display
# Jalankan: streamlit run streamlit_apps/p13_demo/02_teks_display.py
# ============================================================
import streamlit as st

st.set_page_config(page_title="Teks & Display", page_icon="📝", layout="wide")

st.title("📝 Komponen Teks & Display")
st.divider()

# ── Hierarki judul ─────────────────────────────────────────
st.header("1. Hierarki Judul")
col1, col2 = st.columns(2)
with col1:
    st.title("st.title()")
    st.header("st.header()")
    st.subheader("st.subheader()")
with col2:
    st.markdown("**st.markdown()** — *italic*, `code`, [link](https://streamlit.io)")
    st.write("st.write() menampilkan: teks, angka, DataFrame, chart, markdown")
    st.text("st.text() — plain monospace\ncocok untuk raw output")
    st.code("x = [i**2 for i in range(5)]\nprint(x)", language="python")

st.divider()

# ── Alert messages ─────────────────────────────────────────
st.header("2. Alert & Status")
col1, col2 = st.columns(2)
with col1:
    st.success("✅ Model berhasil dilatih! Akurasi: 94.2%")
    st.error("❌ File tidak ditemukan. Cek path-nya.")
with col2:
    st.warning("⚠️ Jumlah data terlalu sedikit untuk cross-validation.")
    st.info("ℹ️ Model menggunakan 80% data untuk training.")

st.divider()

# ── Metrics ───────────────────────────────────────────────
st.header("3. Metrics — KPI Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi Model", "94.2%",  "+2.1%")
col2.metric("Jumlah Data",   "10,482", "+1,203")
col3.metric("F1-Score",      "0.93",   "-0.01",  delta_color="inverse")
col4.metric("Waktu Prediksi","12 ms",  "-3 ms",   delta_color="inverse")
