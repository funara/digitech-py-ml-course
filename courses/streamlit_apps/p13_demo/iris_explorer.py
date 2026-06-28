# ============================================================
# Demo Lengkap — Iris EDA Explorer
# Menggabungkan: sidebar, widgets, columns, tabs, charts
# Jalankan: streamlit run streamlit_apps/p13_demo/iris_explorer.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Konfigurasi ───────────────────────────────────────────
st.set_page_config(
    page_title="Iris EDA Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNA = {'setosa': '#3498db', 'versicolor': '#e74c3c', 'virginica': '#2ecc71'}

# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df, iris

df, iris_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🌸 Iris Explorer")
    st.caption("Dataset klasik 150 sampel bunga Iris")
    st.divider()

    spesies_pilihan = st.multiselect(
        "Filter Spesies",
        options=sorted(df['species'].unique()),
        default=sorted(df['species'].unique()),
    )

    st.divider()

    col_fitur = [c for c in df.columns if c != 'species']
    fitur_x = st.selectbox("Sumbu X (Scatter Plot)", col_fitur, index=2)
    fitur_y = st.selectbox("Sumbu Y (Scatter Plot)", col_fitur, index=3)

    st.divider()

    tampilkan_tabel = st.checkbox("Tampilkan Tabel Data", value=False)
    tampilkan_stats = st.checkbox("Tampilkan Statistik Deskriptif", value=True)

# Filter data berdasarkan sidebar
if not spesies_pilihan:
    st.error("⚠️ Pilih minimal satu spesies di sidebar!")
    st.stop()

df_f = df[df['species'].isin(spesies_pilihan)]

# ── Header ────────────────────────────────────────────────
st.title("🌸 Iris Dataset — EDA Explorer")
st.markdown(f"Menampilkan **{len(df_f):,}** dari {len(df):,} sampel · "
            f"Filter: {', '.join(spesies_pilihan)}")
st.divider()

# ── Metrics row ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sampel",  len(df_f),                  f"{len(df_f)-150}")
c2.metric("Fitur",         len(col_fitur))
c3.metric("Spesies",       df_f['species'].nunique())
c4.metric("Missing Values",df_f.isnull().sum().sum())

st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Scatter Plot", "📈 Distribusi", "🤖 Demo Model ML"])

# Tab 1: Scatter Plot
with tab1:
    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sp, grp in df_f.groupby('species'):
            ax.scatter(grp[fitur_x], grp[fitur_y],
                       label=sp.capitalize(), color=WARNA[sp], alpha=0.75, s=60, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(fitur_x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(fitur_y.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{fitur_x} vs {fitur_y}', fontsize=13)
        ax.legend()
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)
    with col_info:
        st.subheader("Korelasi")
        corr = df_f[[fitur_x, fitur_y]].corr().iloc[0, 1]
        st.metric("r Pearson", f"{corr:.3f}")
        if abs(corr) > 0.7:
            st.success("Korelasi kuat")
        elif abs(corr) > 0.4:
            st.warning("Korelasi sedang")
        else:
            st.info("Korelasi lemah")

        if tampilkan_stats:
            st.subheader("Statistik")
            st.dataframe(df_f[[fitur_x, fitur_y]].describe().round(2), use_container_width=True)

# Tab 2: Distribusi
with tab2:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, fitur in zip(axes.flatten(), col_fitur):
        for sp, grp in df_f.groupby('species'):
            ax.hist(grp[fitur], alpha=0.55, label=sp.capitalize(),
                    bins=15, color=WARNA[sp], edgecolor='white')
        ax.set_title(fitur.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('cm')
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
    plt.suptitle('Distribusi Fitur per Spesies', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 3: Demo Model
with tab3:
    st.subheader("🤖 Klasifikasi Iris dengan Random Forest")
    st.markdown("Coba ubah parameter di bawah dan klik **Latih Model** untuk melihat pengaruhnya.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        n_trees  = st.slider("n_estimators (jumlah pohon)", 10, 300, 100, 10)
    with col_p2:
        max_dep  = st.select_slider("max_depth", options=[None, 3, 5, 10, 20], value=None)
    with col_p3:
        ts       = st.slider("Test size (%)", 10, 40, 20, 5) / 100

    if st.button("🚀 Latih Model", type="primary", use_container_width=True):
        with st.spinner("Melatih Random Forest..."):
            X = df[col_fitur].values
            y = iris_raw.target
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
            clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_dep, random_state=42)
            clf.fit(X_tr, y_tr)
            acc_tr = accuracy_score(y_tr, clf.predict(X_tr))
            acc_te = accuracy_score(y_te, clf.predict(X_te))

        st.success("✅ Model selesai dilatih!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Akurasi Train", f"{acc_tr:.2%}")
        m2.metric("Akurasi Test",  f"{acc_te:.2%}", f"{acc_te - acc_tr:.2%}")
        m3.metric("Jumlah Test",   len(y_te))

        # Feature importance
        fi = pd.Series(clf.feature_importances_, index=col_fitur).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3))
        fi.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
        ax.set_title('Feature Importance', fontsize=11)
        ax.set_xlabel('Importance')
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📋 Classification Report (Test Set)"):
            cr = classification_report(y_te, clf.predict(X_te), target_names=iris_raw.target_names)
            st.text(cr)

# ── Tabel Data ────────────────────────────────────────────
if tampilkan_tabel:
    st.divider()
    st.subheader("📋 Data yang Difilter")
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=300)
