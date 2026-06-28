# ============================================================
# Demo 01 — Struktur Dasar Streamlit
# Jalankan: streamlit run streamlit_apps/p13_demo/01_hello.py
# ============================================================
import streamlit as st

# Konfigurasi halaman — SELALU taruh di baris pertama setelah import
st.set_page_config(
    page_title="Hello Streamlit!",
    page_icon="👋",
    layout="centered",   # 'centered' atau 'wide'
)

# ── Judul & Teks ────────────────────────────────────────────
st.title("👋 Hello, Streamlit!")
st.header("Ini adalah Header")
st.subheader("Ini adalah Subheader")

st.write("st.write() bisa tampilkan apa saja: teks, angka, DataFrame, chart!")
st.markdown("**Markdown** juga *didukung* — bisa pakai `code inline`, bullet list, dll.")

# ── Input Sederhana ─────────────────────────────────────────
nama = st.text_input("Siapa nama kamu?", placeholder="Masukkan nama...")

if nama:
    st.success(f"Halo, **{nama}**! Selamat datang di Streamlit 🎉")
else:
    st.info("Masukkan namamu di atas untuk mendapat sapaan!")
