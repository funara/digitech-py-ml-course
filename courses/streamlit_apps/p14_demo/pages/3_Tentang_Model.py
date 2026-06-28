"""
3_Tentang_Model.py — Halaman Performa & Info Model
Pertemuan 14: Deploy ML App dengan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tentang Model — California Housing",
    page_icon="🤖",
    layout="wide",
)

# ── Load & train model (di-cache — shared dengan halaman Prediksi) ─────────────
@st.cache_resource
def train_model():
    """
    Latih model dan kembalikan model + split data untuk evaluasi.
    @st.cache_resource: model di-share antar halaman dan antar user.
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_names

model, X_train, X_test, y_train, y_test, feature_names = train_model()

# ── Hitung metrik performa ─────────────────────────────────────────────────────
@st.cache_data
def compute_metrics(_model, _X_train, _X_test, _y_train, _y_test):
    """Hitung MAE, RMSE, R² untuk train dan test set."""
    y_pred_train = _model.predict(_X_train)
    y_pred_test  = _model.predict(_X_test)

    metrics = {
        'train': {
            'MAE':  mean_absolute_error(_y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(_y_train, y_pred_train)),
            'R2':   r2_score(_y_train, y_pred_train),
        },
        'test': {
            'MAE':  mean_absolute_error(_y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(_y_test, y_pred_test)),
            'R2':   r2_score(_y_test, y_pred_test),
        },
        'y_pred_test': y_pred_test,
    }
    return metrics

metrics = compute_metrics(model, X_train, X_test, y_train, y_test)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 Tentang Model")
st.markdown(
    "Halaman ini menampilkan performa **Random Forest Regressor** "
    "yang digunakan untuk memprediksi harga rumah California."
)

# ── Info training ──────────────────────────────────────────────────────────────
st.subheader("⚙️ Konfigurasi Training")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
with cfg_col1:
    st.info(f"**Algoritma:** Random Forest Regressor")
    st.info(f"**n_estimators:** 100 pohon")
with cfg_col2:
    st.info(f"**max_depth:** 15")
    st.info(f"**min_samples_split:** 5")
with cfg_col3:
    st.info(f"**Training size:** {len(X_train):,} sampel (80%)")
    st.info(f"**Test size:** {len(X_test):,} sampel (20%)")

st.divider()

# ── Metrik performa ────────────────────────────────────────────────────────────
st.subheader("📈 Performa Model")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="MAE (Test)",
        value=f"{metrics['test']['MAE']:.4f} × $100K",
        delta=f"Train: {metrics['train']['MAE']:.4f}",
        delta_color="inverse",
        help="Mean Absolute Error — rata-rata selisih absolut prediksi vs aktual"
    )
    st.caption(f"= ~${metrics['test']['MAE'] * 100_000:,.0f} rata-rata error")

with m2:
    st.metric(
        label="RMSE (Test)",
        value=f"{metrics['test']['RMSE']:.4f} × $100K",
        delta=f"Train: {metrics['train']['RMSE']:.4f}",
        delta_color="inverse",
        help="Root Mean Squared Error — memberi penalti lebih besar untuk error besar"
    )
    st.caption(f"= ~${metrics['test']['RMSE'] * 100_000:,.0f} RMSE")

with m3:
    st.metric(
        label="R² Score (Test)",
        value=f"{metrics['test']['R2']:.4f}",
        delta=f"Train: {metrics['train']['R2']:.4f}",
        help="R² = proporsi variansi target yang dijelaskan model (1.0 = sempurna)"
    )
    st.caption(f"Model menjelaskan {metrics['test']['R2']*100:.1f}% variansi harga")

# Interpretasi metrik
st.markdown("**Interpretasi Metrik:**")
col_interp1, col_interp2 = st.columns(2)
with col_interp1:
    if metrics['test']['R2'] > 0.80:
        st.success(f"R² = {metrics['test']['R2']:.3f} — Model sangat baik (> 0.80)!")
    elif metrics['test']['R2'] > 0.60:
        st.info(f"R² = {metrics['test']['R2']:.3f} — Model cukup baik (0.60–0.80).")
    else:
        st.warning(f"R² = {metrics['test']['R2']:.3f} — Model perlu ditingkatkan (< 0.60).")

with col_interp2:
    gap_r2 = metrics['train']['R2'] - metrics['test']['R2']
    if gap_r2 > 0.10:
        st.warning(f"Gap Train/Test R² = {gap_r2:.3f} — Ada indikasi sedikit overfitting.")
    else:
        st.success(f"Gap Train/Test R² = {gap_r2:.3f} — Model generalisasi dengan baik!")

st.divider()

# ── Actual vs Predicted plot ────────────────────────────────────────────────────
st.subheader("🎯 Aktual vs Prediksi (Test Set)")

# Sample 1000 untuk visualisasi
idx_sample = np.random.RandomState(42).choice(len(y_test), min(1000, len(y_test)), replace=False)
y_actual_sample   = y_test[idx_sample]
y_pred_sample     = metrics['y_pred_test'][idx_sample]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    y_actual_sample, y_pred_sample,
    alpha=0.4, s=15, color='#3498db', label='Prediksi'
)
# Garis ideal (y = x)
lims = [min(y_actual_sample.min(), y_pred_sample.min()),
        max(y_actual_sample.max(), y_pred_sample.max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Prediksi sempurna (y=x)')
ax.set_xlabel("Harga Aktual (×$100K)", fontsize=11)
ax.set_ylabel("Harga Prediksi (×$100K)", fontsize=11)
ax.set_title(
    f"Aktual vs Prediksi — Random Forest\nR² = {metrics['test']['R2']:.4f} | RMSE = {metrics['test']['RMSE']:.4f}",
    fontsize=13
)
ax.legend()
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption(
    "Titik-titik yang mendekati garis merah putus-putus = prediksi akurat. "
    "Penyebaran jauh dari garis = error prediksi."
)

st.divider()

# ── Feature importance ─────────────────────────────────────────────────────────
st.subheader("📊 Feature Importance")
st.markdown(
    "Feature importance menunjukkan kontribusi relatif setiap fitur "
    "dalam pengambilan keputusan model Random Forest."
)

importances = model.feature_importances_
df_imp = pd.DataFrame({
    'Fitur': list(feature_names),
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#3498db' if imp > importances.mean() else '#bdc3c7' for imp in df_imp['Importance']]
bars = ax.barh(df_imp['Fitur'], df_imp['Importance'], color=colors, edgecolor='white')
ax.axvline(importances.mean(), color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Rata-rata importance = {importances.mean():.3f}')
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title("Feature Importance — Random Forest Regressor", fontsize=13)
ax.legend()

for bar, val in zip(bars, df_imp['Importance']):
    ax.text(
        bar.get_width() + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}',
        va='center', fontsize=9
    )

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Tabel deskripsi fitur ─────────────────────────────────────────────────────
st.subheader("📋 Deskripsi Fitur Dataset")

fitur_desc = pd.DataFrame({
    'Fitur': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
              'Population', 'AveOccup', 'Latitude', 'Longitude'],
    'Deskripsi Lengkap': [
        'Median pendapatan rumah tangga dalam blok sensus (dalam kelipatan $10,000)',
        'Median usia rumah dalam blok sensus (tahun)',
        'Rata-rata jumlah kamar total per rumah tangga dalam blok',
        'Rata-rata jumlah kamar tidur per rumah tangga dalam blok',
        'Jumlah total penduduk dalam blok sensus',
        'Rata-rata jumlah penghuni per rumah tangga dalam blok',
        'Garis lintang blok (32° = selatan, 42° = utara California)',
        'Garis bujur blok (-124° = barat pesisir, -114° = timur pedalaman)',
    ],
    'Satuan': [
        '×$10,000', 'Tahun', 'Kamar', 'Kamar tidur',
        'Orang', 'Orang/rumah', 'Derajat', 'Derajat'
    ],
    'Importance': [f"{importances[i]:.3f}" for i in range(len(feature_names))]
})

st.dataframe(fitur_desc, use_container_width=True, hide_index=True)
