# ============================================================
# FICHIER : precompute.py
# RÔLE    : pré-calculer tous les tickers du Nasdaq 100
# Lance UNE SEULE FOIS : python precompute.py
# Durée estimée : 3 à 5 heures
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import time
from pipeline import (preparer_donnees, creer_features, splitter,
                      entrainer_QR, entrainer_QNN,
                      calculer_metriques, calculer_shap)

# ── Charger la liste des tickers ──
df_all  = pd.read_csv("nasdaq100_merged_1.csv")
TICKERS = sorted(df_all["ticker"].unique().tolist())
print(f"Total tickers à calculer : {len(TICKERS)}")

# ── Résumé de progression ──
deja_faits  = [t for t in TICKERS
               if os.path.exists(f"data/cache_{t}.pkl")]
a_calculer  = [t for t in TICKERS
               if not os.path.exists(f"data/cache_{t}.pkl")]

print(f"Déjà calculés  : {len(deja_faits)}")
print(f"Restant        : {len(a_calculer)}")
print(f"Tickers restants : {a_calculer}\n")

if len(a_calculer) == 0:
    print("Tous les tickers sont déjà calculés !")
    exit()

# ── Boucle principale ──
erreurs    = []
debut_global = time.time()

for i, ticker in enumerate(a_calculer):
    debut = time.time()
    print(f"[{i+1}/{len(a_calculer)}] {ticker}...", end=" ", flush=True)

    try:
        df           = preparer_donnees(ticker)

        # Vérifier qu'on a assez de données
        if len(df) < 100:
            print(f"SKIP (seulement {len(df)} jours)")
            erreurs.append((ticker, "pas assez de données"))
            continue

        df           = creer_features(df)
        X_tr, X_te, y_tr, y_te, dates, _ = splitter(df)

        # Vérifier que le split donne assez de données
        if len(X_tr) < 50 or len(X_te) < 20:
            print(f"SKIP (split trop petit)")
            erreurs.append((ticker, "split trop petit"))
            continue

        preds_qr     = entrainer_QR(X_tr, y_tr, X_te)
        preds_qnn, _ = entrainer_QNN(X_tr, y_tr, X_te, epochs=100)
        metriques    = calculer_metriques(y_te, preds_qr, preds_qnn)
        shap_vals    = calculer_shap(X_tr, X_te)

        results = {
            "df"        : df,
            "X_tr"      : X_tr,
            "X_te"      : X_te,
            "y_te"      : y_te,
            "dates"     : dates,
            "preds_qr"  : preds_qr,
            "preds_qnn" : preds_qnn,
            "metriques" : metriques,
            "shap_vals" : shap_vals,
        }

        with open(f"data/cache_{ticker}.pkl", "wb") as f:
            pickle.dump(results, f)

        duree = time.time() - debut
        print(f"OK ({duree:.0f}s)")

    except Exception as e:
        print(f"ERREUR : {e}")
        erreurs.append((ticker, str(e)))

# ── Résumé final ──
duree_totale = time.time() - debut_global
minutes      = int(duree_totale // 60)
secondes     = int(duree_totale % 60)

print(f"\n{'='*50}")
print(f"TERMINE en {minutes}min {secondes}s")
print(f"Tickers calculés : {len(a_calculer) - len(erreurs)}/{len(a_calculer)}")

if erreurs:
    print(f"\nErreurs ({len(erreurs)}) :")
    for t, e in erreurs:
        print(f"  - {t} : {e}")

caches = [f for f in os.listdir("data") if f.startswith("cache_")]
print(f"\nFichiers cache dans data/ : {len(caches)}")
print("Lance maintenant : streamlit run streamlit_app.py")