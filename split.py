# ============================================================
# FICHIER : split.py
# RÔLE    : normalisation + split train/test
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# --- 1. Charger les features ---
df = pd.read_csv("data/nvda_features.csv", parse_dates=["date"])
print(f"Données chargées : {len(df)} lignes")

# --- 2. Définir X (features) et y (cible) ---
FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag5",
            "vol_moy_5j", "vol_moy_10j", "vol_moy_20j"]

X = df[FEATURES].values   # tableau des variables explicatives
y = df["volatilite"].values  # tableau de la cible
dates = df["date"].values    # on garde les dates pour les graphiques

print(f"Nombre de features : {len(FEATURES)}")
print(f"Taille de X : {X.shape}")
print(f"Taille de y : {y.shape}")

# --- 3. Split temporel 80% / 20% ---
split = int(len(df) * 0.80)

X_train = X[:split]
X_test  = X[split:]
y_train = y[:split]
y_test  = y[split:]
dates_train = dates[:split]
dates_test  = dates[split:]

print(f"\nSplit à la ligne : {split}")
print(f"Train : {len(X_train)} jours  ({dates_train[0]} → {dates_train[-1]})")
print(f"Test  : {len(X_test)}  jours  ({dates_test[0]} → {dates_test[-1]})")

# --- 4. Normalisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 5. Vérification ---
print(f"\nAvant normalisation — moyenne lag1 : {X_train[:,0].mean():.4f}")
print(f"Après normalisation — moyenne lag1 : {X_train_scaled[:,0].mean():.4f}")
print(f"Après normalisation — écart-type lag1 : {X_train_scaled[:,0].std():.4f}")

# --- 6. Graphique du split ---
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(dates_train, y_train, color="#534AB7", 
        linewidth=0.6, label=f"Train ({len(y_train)} jours)")
ax.plot(dates_test, y_test, color="#D85A30",   
        linewidth=0.6, label=f"Test ({len(y_test)} jours)")
ax.axvline(x=dates_test[0], color="black", 
           linewidth=1.5, linestyle="--", label="Séparation train/test")
ax.set_title("Split Train / Test — NVDA")
ax.set_ylabel("Volatilité")
ax.legend()

plt.tight_layout()
plt.savefig("figures/03_split_train_test.png", dpi=150)
plt.show()
print("Graphique sauvegardé dans figures/")

# --- 7. Sauvegarder tout ---
np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_test.npy",  X_test_scaled)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy",  y_test)
np.save("data/dates_test.npy", dates_test)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nFichiers sauvegardés dans data/ :")
print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("  dates_test.npy, scaler.pkl")
print("\nPreparation des donnees TERMINEE !")