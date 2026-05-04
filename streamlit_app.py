# ============================================================
# FICHIER : streamlit_app.py — design premium final
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap, warnings
import os, json
from home_stats import charger_stats_globales
warnings.filterwarnings("ignore")

from pipeline import (preparer_donnees, creer_features, splitter,
                      entrainer_QR, entrainer_QNN,
                      calculer_metriques, calculer_shap, FEATURES)

C_BG      = "#080B0F"
C_SURFACE = "#0E1318"
C_CARD    = "#131920"
C_BORDER  = "#1E2730"
C_BORDER2 = "#2A3540"
C_GOLD    = "#C9A84C"
C_GOLD2   = "#E8C97A"
C_TEAL    = "#4ECDC4"
C_RED     = "#E05C5C"
C_BLUE    = "#5B9BD5"
C_TEXT    = "#F0F4F8"
C_MUTED   = "#6B7E8F"
C_DIM     = "#3A4A5A"

st.set_page_config(
    page_title="QNN · Volatility Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background-color: {C_BG}; color: {C_TEXT}; }}
  .stApp {{ background-color: {C_BG}; }}
  section[data-testid="stSidebar"] {{ background-color: {C_SURFACE}; border-right: 1px solid {C_BORDER}; }}
  section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
  .stTabs [data-baseweb="tab-list"] {{ background: transparent; border-bottom: 1px solid {C_BORDER}; gap: 0px; padding: 0; }}
  .stTabs [data-baseweb="tab"] {{ background: transparent; color: {C_MUTED}; border-bottom: 2px solid transparent; border-radius: 0; font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; padding: 12px 24px; margin-bottom: -1px; }}
  .stTabs [aria-selected="true"] {{ background: transparent !important; color: {C_GOLD} !important; border-bottom: 2px solid {C_GOLD} !important; }}
  [data-testid="metric-container"] {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-top: 2px solid {C_GOLD}; border-radius: 0px 0px 6px 6px; padding: 18px 20px; }}
  [data-testid="metric-container"] label {{ color: {C_MUTED} !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; font-weight: 500 !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {C_TEXT} !important; font-family: 'DM Mono', monospace !important; font-size: 22px !important; font-weight: 500 !important; }}
  .stDataFrame {{ border: 1px solid {C_BORDER}; border-radius: 4px; }}
  .stDataFrame th {{ background: {C_CARD} !important; color: {C_MUTED} !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }}
  .stRadio label {{ color: {C_TEXT} !important; font-size: 13px !important; }}
  .stSlider label {{ color: {C_MUTED} !important; font-size: 11px !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.1em !important; }}
  .stSelectbox label {{ color: {C_MUTED} !important; font-size: 10px !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; }}
  .stSelectbox > div > div {{ background: {C_CARD} !important; border: 1px solid {C_BORDER2} !important; border-radius: 4px !important; color: {C_TEXT} !important; }}
  .stInfo, .stSuccess {{ background: {C_CARD} !important; border: 1px solid {C_BORDER2} !important; border-left: 3px solid {C_GOLD} !important; border-radius: 0 4px 4px 0 !important; color: {C_TEXT} !important; font-size: 13px !important; }}
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: {C_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {C_BORDER2}; border-radius: 2px; }}
  h1, h2, h3 {{ font-family: 'Syne', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }}
  .label {{ font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; color: {C_MUTED}; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 6px; }}
  .divider {{ border: none; border-top: 1px solid {C_BORDER}; margin: 24px 0; }}
  .insight-box {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-left: 3px solid {C_GOLD}; padding: 14px 18px; margin-top: 10px; border-radius: 0 4px 4px 0; }}
  .insight-label {{ font-family: 'DM Mono', monospace; font-size: 9px; color: {C_GOLD}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px; font-weight: 500; }}
  .insight-text {{ font-size: 13px; color: {C_TEXT}; line-height: 1.7; font-weight: 300; }}
  .verdict-box {{ background: linear-gradient(135deg, #0A1A0F, #0D1F14); border: 1px solid #1A3A22; border-left: 3px solid {C_TEAL}; padding: 20px 24px; border-radius: 0 6px 6px 0; margin: 16px 0; }}
  .verdict-label {{ font-family: 'DM Mono', monospace; font-size: 9px; color: {C_TEAL}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px; font-weight: 500; }}
  .verdict-title {{ font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: {C_TEXT}; margin-bottom: 6px; }}
  .verdict-sub {{ font-size: 13px; color: {C_MUTED}; font-weight: 300; }}
  .rank-card {{ background: {C_CARD}; border: 1px solid {C_BORDER}; padding: 14px 16px; margin: 6px 0; border-radius: 4px; display: flex; align-items: center; gap: 14px; }}
  .rank-num {{ font-family: 'DM Mono', monospace; font-size: 11px; color: {C_MUTED}; width: 20px; }}
  .rank-name {{ font-family: 'DM Mono', monospace; font-size: 15px; font-weight: 500; color: {C_TEXT}; flex: 1; }}
  .rank-val {{ font-family: 'DM Mono', monospace; font-size: 12px; color: {C_GOLD}; }}
  .arch-badge-yes {{ display: inline-block; background: #0A1A0F; border: 1px solid {C_TEAL}; color: {C_TEAL}; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 4px 12px; border-radius: 2px; }}
  .arch-badge-no {{ display: inline-block; background: #1A0A0A; border: 1px solid {C_MUTED}; color: {C_MUTED}; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 4px 12px; border-radius: 2px; }}
  .arch-badge-red {{ display: inline-block; background: #1A0A0A; border: 1px solid {C_RED}; color: {C_RED}; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 4px 12px; border-radius: 2px; }}
  .arch-row {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 16px 20px; margin-top: 12px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  .arch-stat {{ font-family: 'DM Mono', monospace; font-size: 12px; color: {C_MUTED}; }}
  .arch-stat b {{ color: {C_TEXT}; font-weight: 500; }}
  .stButton > button {{ background: {C_CARD} !important; border: 1px solid {C_GOLD} !important; color: {C_GOLD} !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; border-radius: 2px !important; padding: 8px 16px !important; width: 100% !important; }}
  .stButton > button:hover {{ background: {C_GOLD} !important; color: {C_BG} !important; }}
  .stDownloadButton > button {{ background: {C_TEAL} !important; border: 1px solid {C_TEAL} !important; color: {C_BG} !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; border-radius: 2px !important; width: 100% !important; }}
  .nc-footer {{ text-align: right; padding: 28px 4px 8px; font-family: 'DM Mono', monospace; font-size: 10px; color: {C_DIM}; letter-spacing: 0.15em; border-top: 1px solid {C_BORDER}; margin-top: 40px; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor' : C_CARD, 'axes.facecolor': C_CARD, 'axes.edgecolor': C_BORDER,
    'axes.labelcolor'  : C_MUTED, 'text.color': C_TEXT, 'xtick.color': C_MUTED,
    'ytick.color'      : C_MUTED, 'grid.color': C_BORDER, 'grid.alpha': 0.4,
    'grid.linestyle'   : ':', 'font.family': 'monospace', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'axes.titlecolor': C_TEXT,
    'axes.titlesize'   : 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'xtick.labelsize'  : 8, 'ytick.labelsize': 8,
})

@st.cache_data
def charger_arch_results():
    path = "data/arch_test_all_tickers.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def charger_adf_results():
    path = "data/adf_test_all_tickers.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_arch_all = charger_arch_results()
df_adf_all  = charger_adf_results()

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 28px 0 24px; border-bottom: 1px solid {C_BORDER}; margin-bottom: 20px'>
      <div style='font-family: DM Mono, monospace; font-size: 9px; color: {C_MUTED}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 10px'>Research Platform</div>
      <div style='font-family: Syne, sans-serif; font-size: 22px; font-weight: 800; color: {C_TEXT}; letter-spacing: -0.02em; line-height: 1.2'>Quantile<br>Neural<br>Networks</div>
      <div style='width: 28px; height: 2px; background: {C_GOLD}; margin-top: 12px; border-radius: 1px'></div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def charger_tickers():
        df = pd.read_csv("nasdaq100_merged_1.csv")
        return sorted(df["ticker"].unique().tolist())

    tickers = charger_tickers()
    ticker  = st.selectbox("Instrument", tickers, index=tickers.index("NVDA"))
    epochs  = st.slider("Epoques QNN", 20, 200, 100, 20)

    if df_arch_all is not None:
        row = df_arch_all[df_arch_all["ticker"] == ticker]
        if len(row) > 0:
            arch_ok   = bool(row["arch_detecte"].values[0])
            arch_pval = float(row["lm_pval"].values[0])
            arch_stat = float(row["lm_stat"].values[0])
            badge_cls = "arch-badge-yes" if arch_ok else "arch-badge-no"
            badge_txt = "ARCH detecte" if arch_ok else "Pas d'effets ARCH"
            arch_col  = C_TEAL if arch_ok else C_MUTED
            st.markdown(f"""
            <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid {C_BORDER}'>
              <div class='label' style='margin-bottom: 8px'>Test ARCH-LM</div>
              <span class='{badge_cls}'>{badge_txt}</span>
              <div style='margin-top: 8px; font-family: DM Mono, monospace; font-size: 10px; color: {C_MUTED}; line-height: 1.8'>
                LM stat &nbsp;&nbsp; <b style='color:{C_TEXT}'>{arch_stat:.2f}</b><br>
                p-value &nbsp;&nbsp; <b style='color:{arch_col}'>{arch_pval:.2e}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if df_adf_all is not None:
        row_adf = df_adf_all[df_adf_all["ticker"] == ticker]
        if len(row_adf) > 0:
            adf_ok   = bool(row_adf["stationnaire"].values[0])
            adf_pval = float(row_adf["adf_pval"].values[0])
            adf_stat = float(row_adf["adf_stat"].values[0])
            adf_col   = C_TEAL if adf_ok else C_RED
            adf_badge = "arch-badge-yes" if adf_ok else "arch-badge-red"
            adf_txt   = "Stationnaire" if adf_ok else "Non stationnaire"
            st.markdown(f"""
            <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid {C_BORDER}'>
              <div class='label' style='margin-bottom: 8px'>Test ADF</div>
              <span class='{adf_badge}'>{adf_txt}</span>
              <div style='margin-top: 8px; font-family: DM Mono, monospace; font-size: 10px; color: {C_MUTED}; line-height: 1.8'>
                ADF stat &nbsp; <b style='color:{C_TEXT}'>{adf_stat:.2f}</b><br>
                p-value &nbsp;&nbsp; <b style='color:{adf_col}'>{adf_pval:.2e}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Bouton PDF ──
    st.markdown(f"""
    <div style='margin-top: 20px; padding-top: 16px; border-top: 1px solid {C_BORDER}'>
      <div class='label' style='margin-bottom: 10px'>Rapport PDF</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generer le rapport PDF", use_container_width=True):
        from generate_report import generer_rapport_pdf
        with st.spinner(f"Generation du rapport pour {ticker}..."):
            try:
                path = generer_rapport_pdf(ticker)
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Telecharger le PDF",
                    data=pdf_bytes,
                    file_name=f"rapport_{ticker}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("Rapport genere avec succes !")
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown(f"""
    <div style='margin-top: 24px; padding-top: 20px; border-top: 1px solid {C_BORDER}'>
      <div class='label'>Quantiles</div>
      <div style='margin-top: 10px; font-size: 12px; line-height: 2.2; font-family: DM Mono, monospace; color: {C_MUTED}'>
        <span style='color:{C_TEAL}'>tau = 0.1</span>  &nbsp; Plancher<br>
        <span style='color:{C_GOLD}'>tau = 0.5</span>  &nbsp; Mediane<br>
        <span style='color:{C_RED}'>tau = 0.9</span>   &nbsp; Risque extreme
      </div>
    </div>
    <div style='margin-top: 24px; padding-top: 20px; border-top: 1px solid {C_BORDER}'>
      <div class='label'>Modeles</div>
      <div style='margin-top: 10px; font-size: 12px; line-height: 2.2; font-family: DM Mono, monospace; color: {C_MUTED}'>
        <b style='color:{C_TEXT}'>QR</b>   Regression Quantile<br>
        <b style='color:{C_TEXT}'>QNN</b>  Quantile Neural Net<br>
        <b style='color:{C_TEXT}'>SHAP</b> Explainability XAI
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ──
st.markdown(f"""
<div style='padding: 32px 0 24px; border-bottom: 1px solid {C_BORDER}; margin-bottom: 28px'>
  <div style='font-family: DM Mono, monospace; font-size: 10px; color: {C_MUTED}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 12px'>
    Ecole Mohammadia d'Ingenieurs &nbsp;·&nbsp; Series Chronologiques &nbsp;·&nbsp; 2026
  </div>
  <div style='font-family: Syne, sans-serif; font-size: 38px; font-weight: 800; color: {C_TEXT}; letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 10px'>
    Prediction de Volatilite
  </div>
  <div style='font-size: 14px; color: {C_MUTED}; font-weight: 300; letter-spacing: 0.01em'>
    Regression Quantile &nbsp;vs&nbsp; Quantile Neural Network &nbsp;·&nbsp; Explainability SHAP &nbsp;·&nbsp; Nasdaq 100
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Chargement des statistiques Nasdaq 100..."):
    glob = charger_stats_globales()

stats_glob  = glob["stats"]
vol_30j     = glob["vol_30j"]
rend_1an    = glob["rend_1an"]
vol_heatmap = glob["vol_heatmap"]

st.markdown(f"<div style='margin-bottom:20px'><div class='label'>Vue d'ensemble — Nasdaq 100 &nbsp;·&nbsp; {stats_glob['date_debut']} → {stats_glob['date_fin']}</div></div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Instruments",    f"{stats_glob['n_tickers']}")
k2.metric("Observations",   f"{stats_glob['n_obs']:,}")
k3.metric("Vol. Nasdaq moy",f"{stats_glob['vol_nasdaq']:.4f}")
k4.metric("Meilleur jour",  f"+{stats_glob['meilleur_jour']['rendement']}%",
          f"{stats_glob['meilleur_jour']['ticker']} · {stats_glob['meilleur_jour']['date']}")
k5.metric("Pire jour",      f"{stats_glob['pire_jour']['rendement']}%",
          f"{stats_glob['pire_jour']['ticker']} · {stats_glob['pire_jour']['date']}",
          delta_color="inverse")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

col_v, col_r = st.columns(2)
with col_v:
    st.markdown(f"<div class='label' style='margin-bottom:12px'>Top 10 — Plus volatils (30 derniers jours)</div>", unsafe_allow_html=True)
    top_vol = vol_30j.head(10).copy()
    top_vol["Vol. 30j"] = top_vol["Vol. 30j"].round(5)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(C_CARD)
    colors_v = [C_RED if i < 3 else C_GOLD if i < 6 else C_DIM+"BB" for i in range(len(top_vol))]
    ax.barh(top_vol["Ticker"][::-1], top_vol["Vol. 30j"][::-1], color=colors_v[::-1], height=0.6, edgecolor='none')
    ax.set_title("VOLATILITE MOYENNE 30J", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    ax.tick_params(length=0)
    ax.set_xlabel("Vol. moyenne", fontsize=8)
    for i, (_, row) in enumerate(top_vol[::-1].iterrows()):
        ax.text(row["Vol. 30j"] + max(top_vol["Vol. 30j"])*0.01, i, f"{row['Vol. 30j']:.4f}", va='center', fontsize=7.5, color=C_MUTED)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_r:
    st.markdown(f"<div class='label' style='margin-bottom:12px'>Top 10 — Meilleurs rendements (1 an)</div>", unsafe_allow_html=True)
    top_rend = rend_1an.head(10).copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(C_CARD)
    colors_r = [C_TEAL if i < 3 else C_GOLD if i < 6 else C_DIM+"BB" for i in range(len(top_rend))]
    ax.barh(top_rend["Ticker"][::-1], top_rend["Rendement cumulé 1an"][::-1], color=colors_r[::-1], height=0.6, edgecolor='none')
    ax.set_title("RENDEMENT CUMULE 1 AN (%)", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    ax.tick_params(length=0)
    ax.set_xlabel("Rendement (%)", fontsize=8)
    for i, (_, row) in enumerate(top_rend[::-1].iterrows()):
        ax.text(row["Rendement cumulé 1an"] + 1, i, f"{row['Rendement cumulé 1an']:.1f}%", va='center', fontsize=7.5, color=C_MUTED)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(f"<div class='label' style='margin-bottom:12px'>Heatmap volatilite moyenne — 100 instruments</div>", unsafe_allow_html=True)
vh       = vol_heatmap.sort_values("vol_moy", ascending=False)
n_cols_h = 10
n_rows_h = int(np.ceil(len(vh) / n_cols_h))
matrix   = np.full((n_rows_h, n_cols_h), np.nan)
for i, val in enumerate(vh["vol_moy"].values):
    matrix[i // n_cols_h, i % n_cols_h] = val
fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor(C_CARD)
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
for i, t in enumerate(vh["ticker"].values):
    ax.text(i % n_cols_h, i // n_cols_h, t, ha='center', va='center', fontsize=6.5, color='black', fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("VOLATILITE MOYENNE PAR INSTRUMENT — rouge = plus volatile", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if df_arch_all is not None:
    st.markdown(f"<div class='label' style='margin-bottom:14px'>Test ARCH-LM — Justification statistique sur le Nasdaq 100</div>", unsafe_allow_html=True)
    n_arch  = int(df_arch_all["arch_detecte"].sum())
    n_total = len(df_arch_all)
    pct     = round(n_arch / n_total * 100, 1)
    a1, a2, a3 = st.columns(3)
    a1.metric("Tickers testes",       f"{n_total}")
    a2.metric("Effets ARCH detectes", f"{n_arch}/{n_total}")
    a3.metric("Taux de detection",    f"{pct}%")
    col_arch_g, col_txt_g = st.columns([3, 2])
    with col_arch_g:
        top15   = df_arch_all.nlargest(15, "lm_stat")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(C_CARD)
        colors_a = [C_RED if v == 0 else C_GOLD if v < 1e-100 else C_TEAL for v in top15["lm_pval"]]
        ax.barh(top15["ticker"][::-1], top15["lm_stat"][::-1], color=colors_a[::-1], height=0.6, edgecolor='none')
        ax.set_title("TOP 15 — STATISTIQUE LM", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
        ax.set_xlabel("Statistique LM", fontsize=8)
        ax.tick_params(length=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_txt_g:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='insight-box'>
          <div class='insight-label'>Interpretation — Test ARCH-LM</div>
          <div class='insight-text'>
            Le test ARCH-LM detecte l'heteroscedasticite conditionnelle dans les rendements financiers.<br><br>
            <b style='color:{C_GOLD}'>{pct}% des instruments</b> du Nasdaq 100 presentent des effets ARCH significatifs (p &lt; 0.05), confirmant que la volatilite est <b>non-constante dans le temps</b>.<br><br>
            Ce resultat justifie l'utilisation du <b>QNN</b> en alternative aux modeles GARCH classiques.
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"<div class='label' style='margin-top:20px; margin-bottom:10px'>Resultats complets — 100 instruments</div>", unsafe_allow_html=True)
    df_arch_disp = df_arch_all[["ticker", "n_obs", "lm_stat", "lm_pval", "arch_detecte"]].copy()
    df_arch_disp.columns = ["Ticker", "Observations", "LM Stat", "P-value", "ARCH detecte"]
    df_arch_disp["ARCH detecte"] = df_arch_disp["ARCH detecte"].map({True: "Oui", False: "Non"})
    df_arch_disp["LM Stat"] = df_arch_disp["LM Stat"].round(2)
    df_arch_disp["P-value"] = df_arch_disp["P-value"].apply(lambda x: f"{x:.2e}")
    st.dataframe(df_arch_disp, use_container_width=True, hide_index=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if df_adf_all is not None:
    st.markdown(f"<div class='label' style='margin-bottom:14px'>Test ADF — Stationnarite des rendements</div>", unsafe_allow_html=True)
    n_stat = int(df_adf_all["stationnaire"].sum())
    n_tot  = len(df_adf_all)
    pct_s  = round(n_stat / n_tot * 100, 1)
    b1, b2, b3 = st.columns(3)
    b1.metric("Tickers testes",     f"{n_tot}")
    b2.metric("Rendements stat.",    f"{n_stat}/{n_tot}")
    b3.metric("Taux stationnarite", f"{pct_s}%")
    st.markdown(f"""
    <div class='insight-box' style='margin-top:12px'>
      <div class='insight-label'>Interpretation — Test ADF</div>
      <div class='insight-text'>
        Le test ADF (Augmented Dickey-Fuller) teste la presence d'une racine unitaire dans les rendements.
        <b style='color:{C_GOLD}'>{pct_s}% des instruments</b> presentent des rendements stationnaires (p &lt; 0.05),
        confirmant que les rendements log sont appropries pour la modelisation directe.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run(ticker, epochs):
    import pickle
    cache_path = f"data/cache_{ticker}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            r = pickle.load(f)
        return (r["df"], r["X_tr"], r["X_te"], r["y_te"], r["dates"],
                r["preds_qr"], r["preds_qnn"], r["metriques"], r["shap_vals"])
    df_t         = preparer_donnees(ticker)
    df_t         = creer_features(df_t)
    X_tr, X_te, y_tr, y_te, dates, _ = splitter(df_t)
    preds_qr     = entrainer_QR(X_tr, y_tr, X_te)
    preds_qnn, _ = entrainer_QNN(X_tr, y_tr, X_te, epochs=epochs)
    metriques    = calculer_metriques(y_te, preds_qr, preds_qnn)
    shap_vals    = calculer_shap(X_tr, X_te)
    return df_t, X_tr, X_te, y_te, dates, preds_qr, preds_qnn, metriques, shap_vals

with st.spinner(f"Chargement de {ticker}..."):
    df, X_tr, X_te, y_te, dates, preds_qr, preds_qnn, metriques, shap_vals = run(ticker, epochs)

split_date = str(pd.to_datetime(dates[0]))[:10]
col_q = 'τ' if 'τ' in metriques.columns else 'tau'
col_m = 'Modèle' if 'Modèle' in metriques.columns else 'Modele'
qnn_rmse = metriques[(metriques[col_m]=='QNN') & (metriques[col_q]==0.9)]['RMSE'].values[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Instrument",     ticker)
c2.metric("Observations",   f"{len(df):,}")
c3.metric("Split",          split_date)
c4.metric("Vol. moyenne",   f"{y_te.mean():.5f}")
c5.metric("QNN RMSE t=0.9", f"{qnn_rmse:.5f}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

COULEURS = {0.1: C_TEAL, 0.5: C_GOLD, 0.9: C_RED}
LABELS   = {0.1: "Plancher", 0.5: "Mediane", 0.9: "Risque extreme"}

tab1, tab2, tab3, tab4 = st.tabs(["EXPLORATION", "PREDICTIONS", "COMPARAISON", "SHAP / XAI"])

# ════════ TAB 1 ════════
with tab1:
    st.markdown(f"<div class='label' style='margin-bottom:16px'>Serie temporelle — {ticker}</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), gridspec_kw={'hspace': 0.55})
    fig.patch.set_facecolor(C_CARD)
    axes[0].plot(df["date"], df["close"], color=C_TEAL, linewidth=0.9)
    axes[0].fill_between(df["date"], df["close"], alpha=0.06, color=C_TEAL)
    axes[0].set_title("PRIX DE CLOTURE  ·  " + ticker, fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    pos = df["rendement"] > 0
    axes[1].bar(df["date"][pos],  df["rendement"][pos],  color=C_TEAL, width=1, alpha=0.7)
    axes[1].bar(df["date"][~pos], df["rendement"][~pos], color=C_RED,  width=1, alpha=0.7)
    axes[1].axhline(0, color=C_BORDER2, linewidth=0.8)
    axes[1].set_title("RENDEMENTS LOGARITHMIQUES", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    axes[2].plot(df["date"], df["volatilite"], color=C_GOLD, linewidth=0.6, alpha=0.9)
    axes[2].fill_between(df["date"], df["volatilite"], alpha=0.1, color=C_GOLD)
    axes[2].set_title("VOLATILITE  ·  |rendement|", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    for ax in axes:
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color(C_BORDER)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"<div class='label' style='margin-top:20px; margin-bottom:12px'>Tests statistiques — {ticker}</div>", unsafe_allow_html=True)
    col_arch_t, col_adf_t = st.columns(2)

    with col_arch_t:
        if df_arch_all is not None:
            row_t = df_arch_all[df_arch_all["ticker"] == ticker]
            if len(row_t) > 0:
                arch_ok   = bool(row_t["arch_detecte"].values[0])
                arch_pval = float(row_t["lm_pval"].values[0])
                arch_stat = float(row_t["lm_stat"].values[0])
                arch_col  = C_TEAL if arch_ok else C_MUTED
                badge_cls = "arch-badge-yes" if arch_ok else "arch-badge-no"
                badge_txt = "Effets ARCH detectes" if arch_ok else "Pas d'effets ARCH"
                arch_interp = (
                    f"La variance des rendements de <b>{ticker}</b> est conditionnellement heteroscedastique "
                    f"(LM = <b>{arch_stat:.2f}</b>, p = <b style='color:{C_TEAL}'>{arch_pval:.2e}</b>). "
                    "Phenomene de <b>volatility clustering</b> confirme — justifie l'utilisation du QNN."
                ) if arch_ok else (
                    f"Pas d'effets ARCH significatifs pour <b>{ticker}</b> "
                    f"(LM = <b>{arch_stat:.2f}</b>, p = <b>{arch_pval:.2e}</b>)."
                )
                st.markdown(f"""
                <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-left:3px solid {arch_col}; border-radius:0 6px 6px 0; padding:16px 18px'>
                  <div style='font-family:DM Mono,monospace; font-size:9px; color:{arch_col}; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:10px; font-weight:600'>Test ARCH-LM</div>
                  <span class='{badge_cls}'>{badge_txt}</span>
                  <div style='margin-top:10px; font-family:DM Mono,monospace; font-size:11px; color:{C_MUTED}; line-height:2'>
                    LM stat &nbsp; <b style='color:{C_TEXT}'>{arch_stat:.2f}</b><br>
                    p-value &nbsp; <b style='color:{arch_col}'>{arch_pval:.2e}</b>
                  </div>
                  <div style='margin-top:10px; font-size:12px; color:{C_MUTED}; line-height:1.6; font-weight:300'>{arch_interp}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_adf_t:
        if df_adf_all is not None:
            row_adf_t = df_adf_all[df_adf_all["ticker"] == ticker]
            if len(row_adf_t) > 0:
                adf_ok   = bool(row_adf_t["stationnaire"].values[0])
                adf_pval = float(row_adf_t["adf_pval"].values[0])
                adf_stat = float(row_adf_t["adf_stat"].values[0])
                adf_col  = C_TEAL if adf_ok else C_RED
                adf_badge_cls = "arch-badge-yes" if adf_ok else "arch-badge-red"
                adf_badge_txt = "Stationnaire" if adf_ok else "Non stationnaire"
                adf_interp = (
                    f"Les rendements de <b>{ticker}</b> sont stationnaires "
                    f"(ADF = <b>{adf_stat:.2f}</b>, p = <b style='color:{C_TEAL}'>{adf_pval:.2e}</b>). "
                    "H0 de racine unitaire rejetee — <b>modelisation directe validee</b>."
                ) if adf_ok else (
                    f"Racine unitaire detectee pour <b>{ticker}</b> "
                    f"(ADF = <b>{adf_stat:.2f}</b>, p = <b style='color:{C_RED}'>{adf_pval:.2e}</b>)."
                )
                st.markdown(f"""
                <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-left:3px solid {adf_col}; border-radius:0 6px 6px 0; padding:16px 18px'>
                  <div style='font-family:DM Mono,monospace; font-size:9px; color:{adf_col}; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:10px; font-weight:600'>Test ADF — Stationnarite</div>
                  <span class='{adf_badge_cls}'>{adf_badge_txt}</span>
                  <div style='margin-top:10px; font-family:DM Mono,monospace; font-size:11px; color:{C_MUTED}; line-height:2'>
                    ADF stat &nbsp; <b style='color:{C_TEXT}'>{adf_stat:.2f}</b><br>
                    p-value &nbsp;&nbsp; <b style='color:{adf_col}'>{adf_pval:.2e}</b>
                  </div>
                  <div style='margin-top:10px; font-size:12px; color:{C_MUTED}; line-height:1.6; font-weight:300'>{adf_interp}</div>
                </div>
                """, unsafe_allow_html=True)

    vol_max_date = df.loc[df["volatilite"].idxmax(), "date"]
    skew = df["rendement"].skew()
    kurt = df["rendement"].kurtosis()
    st.markdown(f"""
    <div class='insight-box' style='margin-top:16px'>
      <div class='insight-label'>Statistiques de la serie</div>
      <div class='insight-text'>
        La serie <b>{ticker}</b> couvre <b>{len(df):,} jours</b> de donnees. La volatilite maximale a ete enregistree le <b>{str(vol_max_date)[:10]}</b>.
        Skewness : <b>{skew:.3f}</b> {'(asymetrie negative)' if skew < 0 else '(asymetrie positive)'} | Kurtosis : <b>{kurt:.1f}</b> {'(queues epaisses — evenements extremes frequents)' if kurt > 3 else '(distribution proche normale)'}.
        Ces caracteristiques justifient l'utilisation de la regression quantile plutot que des methodes basees sur la moyenne.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='label' style='margin-top:24px; margin-bottom:12px'>Statistiques descriptives</div>", unsafe_allow_html=True)
    stats_desc = df["volatilite"].describe().round(5)
    st.dataframe(pd.DataFrame({
        "Statistique": ["N", "Moyenne", "Std", "Min", "Q10", "Q25", "Mediane", "Q75", "Q90", "Max"],
        "Valeur": [
            f"{int(stats_desc['count']):,}", f"{stats_desc['mean']:.5f}", f"{stats_desc['std']:.5f}",
            f"{stats_desc['min']:.5f}", f"{df['volatilite'].quantile(0.10):.5f}", f"{stats_desc['25%']:.5f}",
            f"{stats_desc['50%']:.5f}", f"{stats_desc['75%']:.5f}", f"{df['volatilite'].quantile(0.90):.5f}",
            f"{stats_desc['max']:.5f}",
        ]
    }), use_container_width=True, hide_index=True)

# ════════ TAB 2 ════════
with tab2:
    col_l, col_r_t2 = st.columns([2, 3])
    with col_l:
        modele_choisi = st.radio("Modele", ["Regression Quantile (QR)", "Quantile Neural Network (QNN)"], horizontal=False)
    preds = preds_qr if "QR" in modele_choisi else preds_qnn
    nom   = "QR" if "QR" in modele_choisi else "QNN"
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), gridspec_kw={'hspace': 0.6})
    fig.patch.set_facecolor(C_CARD)
    for i, tau in enumerate([0.1, 0.5, 0.9]):
        axes[i].fill_between(dates, y_te, alpha=0.08, color=C_DIM)
        axes[i].plot(dates, y_te, color=C_DIM, linewidth=0.5, alpha=0.6, label="Reel")
        axes[i].plot(dates, preds[tau], color=COULEURS[tau], linewidth=1.5, label=f"{nom}  tau={tau}", zorder=3)
        axes[i].fill_between(dates, preds[tau], alpha=0.1, color=COULEURS[tau])
        axes[i].set_ylabel("Volatilite", fontsize=8)
        axes[i].legend(loc="upper right", fontsize=8, facecolor=C_SURFACE, edgecolor=C_BORDER, framealpha=0.8)
        axes[i].set_title(f"TAU = {tau}  ·  {LABELS[tau].upper()}", fontsize=9, fontweight='bold', color=COULEURS[tau], loc='left')
        axes[i].tick_params(length=0)
    fig.suptitle(f"{modele_choisi.upper()}  ·  {ticker}", fontsize=10, fontweight='bold', color=C_MUTED, y=1.01, x=0.02, ha='left')
    st.pyplot(fig, use_container_width=True)
    plt.close()
    metr_09  = metriques[metriques[col_q]==0.9]
    rmse_nom = metr_09[metr_09[col_m]==nom]['RMSE'].values[0]
    st.markdown(f"""
    <div class='insight-box'>
      <div class='insight-label'>Interpretation — {nom}</div>
      <div class='insight-text'>
        {'Le modele de regression quantile lineaire estime directement les quantiles de la distribution conditionnelle. Il est rapide et bien calibre sur les quantiles intermediaires. Sa limite est l incapacite a capturer les relations non-lineaires pour tau=0.9 lors des crises.'
         if nom == 'QR' else
         'Le QNN exploite des transformations non-lineaires (ReLU) pour capturer des dynamiques complexes. Pour tau=0.9, il reagit mieux aux clusters de forte volatilite. RMSE = ' + str(round(rmse_nom,5)) + ' sur la periode de test.'}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════ TAB 3 ════════
with tab3:
    st.markdown(f"<div class='label' style='margin-bottom:14px'>Metriques comparatives</div>", unsafe_allow_html=True)
    st.dataframe(metriques, use_container_width=True, hide_index=True)
    st.markdown(f"<div class='label' style='margin-top:24px; margin-bottom:14px'>Zoom tau = 0.9 — Risque extreme</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(C_CARD)
    ax.fill_between(dates, y_te, alpha=0.06, color=C_DIM)
    ax.plot(dates, y_te, color=C_DIM, linewidth=0.5, alpha=0.5, label="Reel", zorder=1)
    ax.plot(dates, preds_qr[0.9], color=C_BLUE, linewidth=1.4, label="QR  tau=0.9", alpha=0.85, zorder=2)
    ax.plot(dates, preds_qnn[0.9], color=C_RED, linewidth=1.4, label="QNN  tau=0.9", alpha=0.85, zorder=3)
    ax.fill_between(dates, preds_qr[0.9], preds_qnn[0.9], alpha=0.06, color=C_GOLD)
    ax.legend(facecolor=C_SURFACE, edgecolor=C_BORDER, fontsize=9, framealpha=0.9)
    ax.set_title(f"QR  vs  QNN  ·  tau=0.9  ·  {ticker}", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
    ax.tick_params(length=0)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    qr_rmse  = metriques[(metriques[col_m]=='QR') & (metriques[col_q]==0.9)]['RMSE'].values[0]
    qnn_rmse = metriques[(metriques[col_m]=='QNN')& (metriques[col_q]==0.9)]['RMSE'].values[0]
    gagnant  = "QNN" if qnn_rmse < qr_rmse else "QR"
    gain     = abs(qr_rmse - qnn_rmse) / qr_rmse * 100
    st.markdown(f"""
    <div class='verdict-box'>
      <div class='verdict-label'>Verdict  ·  tau = 0.9</div>
      <div class='verdict-title'>{gagnant} remporte le quantile extreme</div>
      <div class='verdict-sub'>
        Amelioration RMSE de <b style='color:{C_TEAL}'>{gain:.1f}%</b> sur tau=0.9 — le plus pertinent pour la gestion du risque.
        {'Le QNN capture des non-linearites lors des clusters de forte volatilite.' if gagnant == 'QNN' else 'La regression quantile presente une meilleure calibration sur ce ticker.'}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='label' style='margin-top:24px; margin-bottom:14px'>Calibration empirique</div>", unsafe_allow_html=True)
    rows_cal = []
    for tau in [0.1, 0.5, 0.9]:
        for n, p in [("QR", preds_qr), ("QNN", preds_qnn)]:
            couv = np.mean(y_te <= p[tau]) * 100
            rows_cal.append({"Modele": n, "tau": tau, "Attendu (%)": tau*100, "Observe (%)": round(couv, 1), "Ecart (pts)": round(couv - tau*100, 1)})
    st.dataframe(pd.DataFrame(rows_cal), use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div class='insight-box'>
      <div class='insight-label'>Interpretation — Calibration</div>
      <div class='insight-text'>
        Un modele parfaitement calibre verrait la volatilite reelle depasser tau=0.9 exactement 10% du temps.
        La QR presente une meilleure calibration theorique car elle minimise directement la pinball loss.
        Le QNN peut presenter de legers ecarts en echange d'une meilleure precision sur les extremes.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════ TAB 4 ════════
with tab4:
    importance = np.abs(shap_vals).mean(axis=0)
    indices    = np.argsort(importance)[::-1]
    st.markdown(f"<div class='label' style='margin-bottom:16px'>Importance globale des variables — QNN tau=0.9</div>", unsafe_allow_html=True)
    col_g, col_r_t4 = st.columns([3, 2])
    with col_g:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(C_CARD)
        bar_colors = [C_GOLD if i == 0 else C_TEAL if i == 1 else C_DIM+"CC" for i in range(len(FEATURES))]
        bars = ax.barh([FEATURES[i] for i in indices[::-1]], importance[indices[::-1]], color=bar_colors[::-1], height=0.5, edgecolor='none')
        for bar, val in zip(bars, importance[indices[::-1]]):
            ax.text(val + max(importance)*0.01, bar.get_y() + bar.get_height()/2, f"{val:.5f}", va='center', ha='left', fontsize=8, color=C_MUTED)
        ax.set_xlabel("Importance SHAP moyenne (|valeur|)", fontsize=8)
        ax.set_title("SHAP FEATURE IMPORTANCE", fontsize=9, fontweight='bold', color=C_MUTED, loc='left')
        ax.tick_params(length=0)
        ax.set_xlim(0, max(importance)*1.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_r_t4:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for rank, idx in enumerate(indices[:5]):
            medal_color = [C_GOLD, C_MUTED, C_DIM, C_DIM, C_DIM][rank]
            rank_label  = ["01", "02", "03", "04", "05"][rank]
            st.markdown(f"""
            <div class='rank-card'>
              <span class='rank-num' style='color:{medal_color}'>{rank_label}</span>
              <span class='rank-name'>{FEATURES[idx]}</span>
              <span class='rank-val'>{importance[idx]:.5f}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown(f"<div class='label' style='margin-top:28px; margin-bottom:14px'>SHAP Summary Plot — Distribution des contributions</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig2.patch.set_facecolor(C_CARD)
    shap.summary_plot(shap_vals, X_te, feature_names=FEATURES, show=False, plot_size=None, color_bar=True)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
    top3 = [FEATURES[indices[i]] for i in range(3)]
    st.markdown(f"""
    <div class='insight-box'>
      <div class='insight-label'>Interpretation SHAP</div>
      <div class='insight-text'>
        Les trois variables les plus influentes sont <b style='color:{C_GOLD}'>{top3[0]}</b>, <b style='color:{C_TEAL}'>{top3[1]}</b> et <b style='color:{C_TEXT}'>{top3[2]}</b>.
        Les valeurs elevees de ces variables <b>augmentent</b> la prediction de volatilite — coherent avec le <b>clustering de volatilite</b>.
        La predominance de {'la memoire longue (moyennes mobiles)' if 'moy' in top3[0] else 'la memoire courte (lags recents)'} suggere que le QNN capte {'un regime de volatilite persistant sur plusieurs semaines.' if 'moy' in top3[0] else 'une dependance forte aux mouvements recents.'}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class='nc-footer'>
  QNN Research Dashboard &nbsp;·&nbsp; Series Chronologiques &nbsp;·&nbsp; EMI 2026 &nbsp;·&nbsp;
  <span style='color:{C_GOLD}; letter-spacing: 0.2em'>N.C.</span>
</div>
""", unsafe_allow_html=True)