"""
1_EDA.py — Halaman Exploratory Data Analysis
Pertemuan 14: Deploy ML App dengan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA — California Housing",
    page_icon="📊",
    layout="wide",
)

# ── Load data dengan cache ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load California Housing dataset — di-cache agar tidak reload setiap interaksi."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Exploratory Data Analysis")
st.markdown("Eksplorasi dataset **California Housing** — distribusi, korelasi, dan peta geografis harga rumah.")

# ── Sidebar: Filter ───────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filter Data")
st.sidebar.markdown("Filter dataset berdasarkan rentang median pendapatan:")

inc_min = float(df['MedInc'].min())
inc_max = float(df['MedInc'].max())

income_range = st.sidebar.slider(
    "Median Income (×$10K)",
    min_value=inc_min,
    max_value=inc_max,
    value=(inc_min, inc_max),
    step=0.1,
    help="Filter blok sensus berdasarkan rentang median pendapatan"
)

# Terapkan filter
df_filtered = df[
    (df['MedInc'] >= income_range[0]) &
    (df['MedInc'] <= income_range[1])
].copy()

st.sidebar.markdown(f"**Data setelah filter:** {len(df_filtered):,} dari {len(df):,} blok")

# ── Metrics ringkas ───────────────────────────────────────────────────────────
st.subheader("Statistik Ringkas (Data Terfilter)")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Rata-rata Harga", f"${df_filtered['MedHouseVal'].mean() * 100_000:,.0f}")
with m2:
    st.metric("Median Harga", f"${df_filtered['MedHouseVal'].median() * 100_000:,.0f}")
with m3:
    st.metric("Harga Tertinggi", f"${df_filtered['MedHouseVal'].max() * 100_000:,.0f}")
with m4:
    st.metric("Harga Terendah", f"${df_filtered['MedHouseVal'].min() * 100_000:,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Distribusi Target", "🔗 Korelasi Fitur", "🗺️ Peta Geografis"])

# ── Tab 1: Distribusi target ──────────────────────────────────────────────────
with tab1:
    st.subheader("Distribusi Harga Rumah (MedHouseVal)")

    col_hist, col_box = st.columns([2, 1])

    with col_hist:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            df_filtered['MedHouseVal'] * 100_000,
            bins=50,
            color='#3498db',
            edgecolor='white',
            alpha=0.85
        )
        ax.axvline(
            df_filtered['MedHouseVal'].mean() * 100_000,
            color='#e74c3c', linestyle='--', linewidth=2,
            label=f"Rata-rata: ${df_filtered['MedHouseVal'].mean() * 100_000:,.0f}"
        )
        ax.axvline(
            df_filtered['MedHouseVal'].median() * 100_000,
            color='#f39c12', linestyle='--', linewidth=2,
            label=f"Median: ${df_filtered['MedHouseVal'].median() * 100_000:,.0f}"
        )
        ax.set_xlabel("Harga Rumah (USD)", fontsize=11)
        ax.set_ylabel("Jumlah Blok Sensus", fontsize=11)
        ax.set_title("Distribusi Harga Median Rumah", fontsize=13)
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_box:
        st.markdown("**Statistik Deskriptif:**")
        desc = df_filtered['MedHouseVal'].describe() * 100_000
        desc.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        desc_df = desc.reset_index()
        desc_df.columns = ['Statistik', 'Harga (USD)']
        desc_df['Harga (USD)'] = desc_df['Harga (USD)'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            > **Catatan:** Banyak nilai di atas $500,000 kemungkinan 
            > merupakan nilai yang di-cap (di-ceiling) oleh dataset.
            """
        )

# ── Tab 2: Correlation heatmap ────────────────────────────────────────────────
with tab2:
    st.subheader("Korelasi Antar Fitur")
    st.markdown(
        "Heatmap korelasi Pearson menunjukkan seberapa kuat hubungan linear "
        "antar fitur. Nilai mendekati **+1** atau **-1** = korelasi kuat."
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    corr_matrix = df_filtered.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title("Correlation Heatmap — California Housing", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Temuan utama
    st.subheader("Temuan Utama")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.success(
            "**MedInc** memiliki korelasi positif tertinggi dengan harga rumah (+0.69) — "
            "semakin tinggi pendapatan median, semakin mahal harga rumah."
        )
    with col_f2:
        st.info(
            "**Latitude** dan **Longitude** berkorelasi negatif kuat dengan harga — "
            "wilayah selatan (Latitude rendah) dan pesisir (Longitude tinggi) lebih mahal."
        )

# ── Tab 3: Geographic scatter ─────────────────────────────────────────────────
with tab3:
    st.subheader("Peta Geografis Harga Rumah California")
    st.markdown(
        "Setiap titik adalah satu blok sensus. "
        "**Warna** menunjukkan harga: 🔵 murah → 🔴 mahal."
    )

    # Sample untuk performa (max 5000 titik)
    sample_size = min(5000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_sample['Longitude'],
        df_sample['Latitude'],
        c=df_sample['MedHouseVal'],
        cmap='RdYlBu_r',
        alpha=0.5,
        s=10,
        vmin=df_filtered['MedHouseVal'].quantile(0.05),
        vmax=df_filtered['MedHouseVal'].quantile(0.95),
    )
    plt.colorbar(scatter, ax=ax, label='Harga Rumah (×$100K)', shrink=0.8)

    # Anotasi kota besar
    kota = {
        'San Francisco': (-122.4, 37.8),
        'Los Angeles':   (-118.2, 34.1),
        'San Diego':     (-117.1, 32.7),
        'Sacramento':    (-121.5, 38.6),
    }
    for nama, (lon, lat) in kota.items():
        ax.annotate(
            nama,
            xy=(lon, lat),
            fontsize=9,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(
        f"Peta Harga Rumah California ({sample_size:,} blok sampel)\n"
        "Merah = Mahal | Biru = Murah",
        fontsize=13
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"Menampilkan {sample_size:,} dari {len(df_filtered):,} blok sensus "
        "(disample untuk performa)."
    )
