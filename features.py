import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Charger les données propres ---
df = pd.read_csv("data/nvda_clean.csv", parse_dates=["date"])
print(f"Données chargées : {len(df)} lignes")

# --- 2. Créer les lags de volatilité ---
# lag1 = volatilité d'hier, lag2 = avant-hier, etc.
for i in range(1, 6):
    df[f"lag{i}"] = df["volatilite"].shift(i)
# --- 3. Créer les volatilités moyennes passées ---
# moyenne mobile sur 5, 10 et 20 jours passés
df["vol_moy_5j"]  = df["volatilite"].shift(1).rolling(5).mean()
df["vol_moy_10j"] = df["volatilite"].shift(1).rolling(10).mean()
df["vol_moy_20j"] = df["volatilite"].shift(1).rolling(20).mean()

# --- 4. Supprimer les lignes incomplètes ---
df = df.dropna().reset_index(drop=True)
print(f"Après création des features : {len(df)} lignes")

# --- 5. Afficher un aperçu ---
print("\nAperçu des features :")
print(df[["date", "volatilite", "lag1", "lag2", 
          "vol_moy_5j", "vol_moy_10j", "vol_moy_20j"]].head(8))

# --- 6. Graphique de vérification ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(df["date"], df["volatilite"], 
             color="#D85A30", linewidth=0.6, label="Volatilité réelle")
axes[0].plot(df["date"], df["vol_moy_5j"],  
             color="#534AB7", linewidth=1, label="Moyenne 5j")
axes[0].plot(df["date"], df["vol_moy_20j"], 
             color="#1D9E75", linewidth=1, label="Moyenne 20j")
axes[0].set_title("Volatilité et moyennes mobiles — NVDA")
axes[0].legend()

axes[1].scatter(df["lag1"], df["volatilite"], 
                alpha=0.2, s=2, color="#534AB7")
axes[1].set_xlabel("Volatilité hier (lag1)")
axes[1].set_ylabel("Volatilité aujourd'hui")
axes[1].set_title("Corrélation : volatilité hier vs aujourd'hui")

plt.tight_layout()
plt.savefig("figures/02_features_nvda.png", dpi=150)
plt.show()
print("Graphique sauvegardé dans figures/")

# --- 7. Sauvegarder ---
df.to_csv("data/nvda_features.csv", index=False)
print("Fichier sauvegardé : data/nvda_features.csv")

# --- 8. Résumé des colonnes créées ---
print("\nColonnes disponibles :")
for col in df.columns:
    print(f"  - {col}")