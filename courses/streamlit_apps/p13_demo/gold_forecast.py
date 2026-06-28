# ============================================================
# Gold Price Forecast — 3 Model Comparison
# Data: yfinance GC=F (Gold Futures)
# Models: RandomForest, Linear Regression, XGBoost
# Jalankan: streamlit run streamlit_apps/gold_forecast.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Gold Price Forecast",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")

# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data(period="2y"):
    import yfinance as yf
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=period, auto_adjust=False)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df

@st.cache_data
def make_features(df, n_lags=5, ma_windows=[5, 10, 20]):
    dfe = df[["date", "close"]].copy()
    for lag in range(1, n_lags + 1):
        dfe[f"lag_{lag}"] = dfe["close"].shift(lag)
    for w in ma_windows:
        dfe[f"ma_{w}"] = dfe["close"].rolling(w).mean()
    dfe["volatility"] = dfe["close"].pct_change().rolling(5).std()
    dfe.dropna(inplace=True)
    return dfe

@st.cache_data
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42),
    }

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🥇 Gold Forecast")
    st.caption("Prediksi harga emas (GC=F) dengan ML")
    st.divider()

    period = st.selectbox("Rentang Data", ["1y", "2y", "5y", "10y", "max"], index=1)
    n_lags = st.slider("Jumlah Lag (hari sebelumnya)", 2, 20, 5)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

    st.divider()
    st.subheader("Model")
    use_lr = st.checkbox("Linear Regression", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    use_xgb = st.checkbox("XGBoost", value=True)

    st.divider()
    tampilkan_tabel = st.checkbox("Tampilkan Tabel Data", value=False)

# ── Load & Feature Engineering ────────────────────────────
df_raw = load_data(period)
df_feat = make_features(df_raw, n_lags=n_lags)

col_fitur = [c for c in df_feat.columns if c not in ("date", "close")]
X = df_feat[col_fitur].values
y = df_feat["close"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, shuffle=False)

# ── Header ────────────────────────────────────────────────
st.title("🥇 Gold Price Forecast — Model Comparison")
st.markdown(f"Data **GC=F** ({period}) · "
            f"Fitur: {len(col_fitur)} · "
            f"Train: {len(X_tr):,} · "
            f"Test: {len(X_te):,}")
st.divider()

# ── Metrics row ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Hari", len(df_feat))
c2.metric("Harga Terakhir", f"${df_feat['close'].iloc[-1]:.2f}")
c3.metric("Harga Tertinggi", f"${df_feat['close'].max():.2f}")
c4.metric("Harga Terendah", f"${df_feat['close'].min():.2f}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Harga & Prediksi", "📊 Perbandingan Model", "⚙️ Detail Model"])

# Tab 1: Price chart & predictions overlay
with tab1:
    # Prepare model predictions
    models_to_run = {}
    if use_lr:
        from sklearn.linear_model import LinearRegression
        models_to_run["Linear Regression"] = LinearRegression()
    if use_rf:
        from sklearn.ensemble import RandomForestRegressor
        models_to_run["Random Forest"] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    if use_xgb:
        from xgboost import XGBRegressor
        models_to_run["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42, verbosity=0)

    if not models_to_run:
        st.warning("Pilih minimal satu model di sidebar.")
        st.stop()

    preds = {}
    with st.spinner("Melatih model..."):
        for name, model in models_to_run.items():
            model.fit(X_tr, y_tr)
            preds[name] = model.predict(X_te)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    dates_te = df_feat["date"].iloc[len(X_tr):].values
    ax.plot(df_feat["date"], df_feat["close"], label="Aktual", color="#2c3e50", linewidth=1.5)

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for idx, (name, y_pred) in enumerate(preds.items()):
        ax.plot(dates_te, y_pred, label=f"{name} (pred)", color=colors[idx % len(colors)],
                linewidth=1.5, linestyle="--", alpha=0.85)

    ax.set_title("Harga Aktual vs Prediksi (Test Set)", fontsize=13)
    ax.set_ylabel("Harga Penutupan (USD)")
    ax.legend()
    sns.despine()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 2: Model comparison metrics
with tab2:
    st.subheader("📊 Perbandingan Performa Model")

    results = []
    for name, y_pred in preds.items():
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        results.append({"Model": name, "MAE": f"${mae:.2f}", "RMSE": f"${rmse:.2f}", "R²": f"{r2:.4f}"})

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.divider()

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics_cfg = [
        ("MAE (USD)", [mean_absolute_error(y_te, v) for v in preds.values()], "#e74c3c"),
        ("RMSE (USD)", [np.sqrt(mean_squared_error(y_te, v)) for v in preds.values()], "#3498db"),
        ("R² Score",   [r2_score(y_te, v) for v in preds.values()], "#2ecc71"),
    ]
    for ax, (title, vals, color) in zip(axes, metrics_cfg):
        bars = ax.bar(preds.keys(), vals, color=color, edgecolor="white", alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticklabels(preds.keys(), rotation=15, fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(vals) * 0.02 if title != "R² Score" else 0.01),
                    f"{v:.2f}" if title != "R² Score" else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
        sns.despine(ax=ax)
    plt.suptitle("Comparison of Model Metrics", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Residuals plot
    st.subheader("Residual Plot")
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4))
    if len(preds) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, preds.items()):
        residuals = y_te - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color="#3498db", edgecolors="white", linewidth=0.3)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 3: Feature importance (for tree-based models)
with tab3:
    st.subheader("⚙️ Detail Model")

    for name, model in models_to_run.items():
        with st.expander(f"**{name}**", expanded=True):
            # Predictions on test set
            y_pred = model.predict(X_te)
            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2 = r2_score(y_te, y_pred)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE", f"${mae:.2f}")
            mc2.metric("RMSE", f"${rmse:.2f}")
            mc3.metric("R²", f"{r2:.4f}")

            # Feature importance if available
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=col_fitur).sort_values()
                fig, ax = plt.subplots(figsize=(8, 3))
                fi.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
                ax.set_title(f"Feature Importance — {name}", fontsize=11)
                ax.set_xlabel("Importance")
                sns.despine(ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if hasattr(model, "coef_"):
                coef = pd.Series(model.coef_, index=col_fitur).sort_values()
                fig, ax = plt.subplots(figsize=(8, 3))
                colors_coef = ["#e74c3c" if v < 0 else "#3498db" for v in coef.values]
                coef.plot(kind="barh", ax=ax, color=colors_coef, edgecolor="white")
                ax.set_title(f"Coefficients — {name}", fontsize=11)
                ax.set_xlabel("Coefficient")
                sns.despine(ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ── Tabel Data ────────────────────────────────────────────
if tampilkan_tabel:
    st.divider()
    st.subheader("📋 Data")
    st.dataframe(df_feat.round(2).reset_index(drop=True), use_container_width=True, height=300)
