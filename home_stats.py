# ============================================================
# FICHIER : home_stats.py
# RÔLE    : KPIs globaux Nasdaq 100 pour la page d'accueil
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def charger_stats_globales():
    """Calcule tous les KPIs globaux une seule fois."""

    df = pd.read_csv("nasdaq100_merged_1.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rendements log par ticker
    df["rendement"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df["volatilite"] = df["rendement"].abs()
    df = df.dropna()

    # ── 1. Volatilité moyenne globale par ticker ──
    vol_moy = (df.groupby("ticker")["volatilite"]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index())
    vol_moy.columns = ["Ticker", "Vol. moyenne"]

    # ── 2. Top volatils sur 30 derniers jours ──
    date_max  = df["date"].max()
    date_30j  = date_max - pd.Timedelta(days=30)
    df_recent = df[df["date"] >= date_30j]
    vol_30j   = (df_recent.groupby("ticker")["volatilite"]
                           .mean()
                           .sort_values(ascending=False)
                           .reset_index())
    vol_30j.columns = ["Ticker", "Vol. 30j"]

    # ── 3. Meilleurs rendements cumulés 1 an ──
    date_1an  = date_max - pd.Timedelta(days=365)
    df_1an    = df[df["date"] >= date_1an]
    rend_1an  = (df_1an.groupby("ticker")["rendement"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index())
    rend_1an.columns  = ["Ticker", "Rendement cumulé 1an"]
    rend_1an["Rendement cumulé 1an"] = (
        rend_1an["Rendement cumulé 1an"] * 100
    ).round(1)

    # ── 4. Meilleur et pire jour tous tickers ──
    df_recent_all = df[df["date"] >= "1990-01-01"]
    idx_min  = df_recent_all["rendement"].idxmin()
    idx_max  = df_recent_all["rendement"].idxmax()
    meilleur = df_recent_all.loc[idx_max, ["ticker", "date", "rendement"]]
    pire     = df_recent_all.loc[idx_min, ["ticker", "date", "rendement"]]

    # ── 5. Stats générales ──
    stats = {
        "n_tickers"    : df["ticker"].nunique(),
        "n_obs"        : len(df),
        "date_debut"   : str(df["date"].min())[:10],
        "date_fin"     : str(df["date"].max())[:10],
        "vol_nasdaq"   : df["volatilite"].mean(),
        "meilleur_jour": {
            "ticker"     : meilleur["ticker"],
            "date"       : str(meilleur["date"])[:10],
            "rendement"  : round(meilleur["rendement"] * 100, 2)
        },
        "pire_jour"    : {
            "ticker"     : pire["ticker"],
            "date"       : str(pire["date"])[:10],
            "rendement"  : round(pire["rendement"] * 100, 2)
        },
    }

    # ── 6. Volatilité par ticker (pour heatmap) ──
    vol_heatmap = (df.groupby("ticker")["volatilite"]
                     .mean()
                     .reset_index())
    vol_heatmap.columns = ["ticker", "vol_moy"]

    return {
        "vol_moy"    : vol_moy,
        "vol_30j"    : vol_30j,
        "rend_1an"   : rend_1an,
        "stats"      : stats,
        "vol_heatmap": vol_heatmap,
        "df_full"    : df,
    }