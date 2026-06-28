# LATIHAN 1: Kalkulator BMI Sederhana (SOLUSI LENGKAP)
# ============================================================
import streamlit as st

st.set_page_config(page_title="Kalkulator BMI", page_icon="⚖️")
st.title("⚖️ Kalkulator BMI")
st.markdown("Masukkan berat dan tinggi badan untuk menghitung BMI")

col1, col2 = st.columns(2)
with col1:
    berat = st.number_input("Berat Badan (kg)", 20.0, 300.0, 65.0, 0.5)
with col2:
    tinggi = st.number_input("Tinggi Badan (cm)", 50.0, 250.0, 165.0, 0.5)

if tinggi > 0:
    bmi = berat / ((tinggi / 100) ** 2)
    st.subheader("Hasil")
    st.metric("BMI Anda", f"{bmi:.1f}")

    if bmi < 18.5:
        st.warning("Kategori: Kurus (Underweight)")
    elif bmi < 25.0:
        st.success("Kategori: Normal (Ideal)")
    elif bmi < 30.0:
        st.warning("Kategori: Gemuk (Overweight)")
    else:
        st.error("Kategori: Obesitas")

    st.progress(min(bmi / 40, 1.0))
else:
    st.info("Masukkan tinggi dan berat badan untuk menghitung BMI")

# ▶️ Jalankan: launch('latihan_1_kalkulator.py')
