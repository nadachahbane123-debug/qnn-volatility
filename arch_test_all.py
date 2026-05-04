# ============================================================
# FICHIER : arch_test_all.py
# RÔLE    : Test ARCH-LM sur tous les tickers du Nasdaq 100
# Durée   : ~2-3 minutes
# ============================================================

import pandas as pd
import numpy as np
import json
from statsmodels.stats.diagnostic import het_arch
import warnings
warnings.filterwarnings("ignore")

# --- Charger tous les tickers ---
df_all  = pd.read_csv("nasdaq100_merged_1.csv")
TICKERS = sorted(df_all["ticker"].unique().tolist())
print(f"Test ARCH-LM sur {len(TICKERS)} tickers...\n")

resultats = []

for i, ticker in enumerate(TICKERS):
    try:
        # Préparer la série
        df = df_all[df_all["ticker"] == ticker].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["rendement"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna()

        # Supprimer aberrations
        seuil = df["rendement"].abs().quantile(0.999)
        df    = df[df["rendement"].abs() < seuil]

        if len(df) < 50:
            print(f"[{i+1}/{len(TICKERS)}] {ticker} — SKIP (pas assez de données)")
            continue

        # Test ARCH-LM
        lm_stat, lm_pval, f_stat, f_pval = het_arch(
            df["rendement"].dropna(), nlags=10
        )

        arch_detecte = bool(lm_pval < 0.05)
        resultats.append({
            "ticker"      : ticker,
            "n_obs"       : len(df),
            "lm_stat"     : round(lm_stat, 4),
            "lm_pval"     : round(lm_pval, 8),
            "f_stat"      : round(f_stat, 4),
            "f_pval"      : round(f_pval, 8),
            "arch_detecte": arch_detecte
        })

        statut = "ARCH detecte" if arch_detecte else "pas d'effets ARCH"
        print(f"[{i+1}/{len(TICKERS)}] {ticker:<6} — {statut:<20} "
              f"(p={lm_pval:.2e})")

    except Exception as e:
        print(f"[{i+1}/{len(TICKERS)}] {ticker} — ERREUR : {e}")

# --- Résumé ---
df_res = pd.DataFrame(resultats)
n_arch = df_res["arch_detecte"].sum()

print(f"\n{'='*50}")
print(f"Tickers testés      : {len(df_res)}")
print(f"Effets ARCH détectés: {n_arch}/{len(df_res)} "
      f"({n_arch/len(df_res)*100:.1f}%)")
print(f"Sans effets ARCH    : {len(df_res)-n_arch}")

# Top 10 plus hétéroscédastiques
print(f"\nTop 10 — LM stat la plus élevée (plus hétéroscédastiques) :")
top10 = df_res.nlargest(10, "lm_stat")[["ticker","lm_stat","lm_pval"]]
print(top10.to_string(index=False))

# --- Sauvegarder ---
df_res.to_csv("data/arch_test_all_tickers.csv", index=False)

with open("data/arch_test_summary.json", "w") as f:
    json.dump({
        "n_tickers_testes" : len(df_res),
        "n_arch_detecte"   : int(n_arch),
        "pct_arch"         : round(n_arch/len(df_res)*100, 1),
        "nvda_pval"        : float(df_res[df_res["ticker"]=="NVDA"]["lm_pval"].values[0])
                             if "NVDA" in df_res["ticker"].values else None
    }, f, indent=2)

print(f"\nRésultats sauvegardés :")
print(f"  data/arch_test_all_tickers.csv")
print(f"  data/arch_test_summary.json")
print(f"\nTest ARCH-LM — TERMINE !")