# ============================================================
# FICHIER : adf_test_all.py
# RÔLE    : Test ADF sur tous les tickers du Nasdaq 100
# Durée   : ~1-2 minutes
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

df_all  = pd.read_csv("nasdaq100_merged_1.csv")
TICKERS = sorted(df_all["ticker"].unique().tolist())
print(f"Test ADF sur {len(TICKERS)} tickers...\n")

resultats = []

for i, ticker in enumerate(TICKERS):
    try:
        df = df_all[df_all["ticker"] == ticker].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["rendement"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna()

        if len(df) < 50:
            print(f"[{i+1}/{len(TICKERS)}] {ticker} — SKIP")
            continue

        adf = adfuller(df["rendement"].dropna(), autolag='AIC')

        stationnaire  = bool(adf[1] < 0.05)
        resultats.append({
            "ticker"      : ticker,
            "n_obs"       : len(df),
            "adf_stat"    : round(adf[0], 4),
            "adf_pval"    : round(adf[1], 8),
            "n_lags"      : adf[2],
            "crit_1pct"   : round(adf[4]["1%"], 4),
            "crit_5pct"   : round(adf[4]["5%"], 4),
            "stationnaire": stationnaire
        })

        statut = "Stationnaire" if stationnaire else "Non stationnaire"
        print(f"[{i+1}/{len(TICKERS)}] {ticker:<6} — {statut:<18} "
              f"(p={adf[1]:.2e})")

    except Exception as e:
        print(f"[{i+1}/{len(TICKERS)}] {ticker} — ERREUR : {e}")

# ── Résumé ──
df_res   = pd.DataFrame(resultats)
n_stat   = df_res["stationnaire"].sum()
n_total  = len(df_res)

print(f"\n{'='*50}")
print(f"Tickers testés     : {n_total}")
print(f"Stationnaires      : {n_stat}/{n_total} ({n_stat/n_total*100:.1f}%)")
print(f"Non stationnaires  : {n_total - n_stat}")

print(f"\nTop 5 — p-values les plus élevées (moins stationnaires) :")
top5 = df_res.nlargest(5, "adf_pval")[["ticker", "adf_stat", "adf_pval"]]
print(top5.to_string(index=False))

df_res.to_csv("data/adf_test_all_tickers.csv", index=False)
print(f"\nRésultats sauvegardés : data/adf_test_all_tickers.csv")
print("Test ADF — TERMINE !")