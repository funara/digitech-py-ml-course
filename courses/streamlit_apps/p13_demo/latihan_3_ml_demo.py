# LATIHAN 3 (Tantangan): Demo Model ML Tim
# Buat app Streamlit yang menampilkan model ML dari dataset tim:
# 1. Upload dataset CSV dengan st.file_uploader
# 2. Sidebar: pilih kolom target, pilih fitur (st.multiselect)
# 3. Pilih model: Logistic Regression, Decision Tree, atau Random Forest
# 4. Klik tombol "Latih Model"
# 5. Tampilkan:
#    a. Akurasi train dan test (st.metric)
#    b. Classification report (st.text)
#    c. Feature importance (bar chart) jika menggunakan tree-based model
# 6. Layout: gunakan st.columns dan st.tabs untuk tampilan yang rapi
# Bonus: tambahkan st.spinner saat model sedang dilatih

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Demo ML Tim", page_icon="🤖", layout="wide")
st.title("🤖 Demo Model ML — Dataset Tim")

# Tulis kode kamu di sini:
