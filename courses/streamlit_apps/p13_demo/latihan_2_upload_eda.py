# LATIHAN 2: EDA Tool untuk Dataset Tim
# Buat app Streamlit yang:
# 1. Ada st.file_uploader untuk upload file CSV
# 2. Setelah upload, tampilkan:
#    a. st.metric untuk: jumlah baris, kolom, missing values
#    b. st.dataframe untuk preview 10 baris pertama
#    c. st.dataframe untuk df.describe() (statistik deskriptif)
# 3. Buat dropdown (st.selectbox) untuk memilih kolom numerik
# 4. Setelah kolom dipilih, tampilkan histogram kolom tersebut dengan st.pyplot
# 5. Bonus: tambahkan correlation heatmap (seaborn) di st.expander

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Tool", page_icon="🔍", layout="wide")
st.title("🔍 EDA Tool — Upload Dataset Timmu")

# Tulis kode kamu di sini:
