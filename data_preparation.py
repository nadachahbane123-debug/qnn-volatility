import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Chargement ---
TICKER = "NVDA"

df = pd.read_csv("nasdaq100_merged_1.csv")

# --- 2. Filtrer NVDA ---
df = df[df["ticker"] == TICKER].copy()

# --- 3. Dates et tri ---
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- 4. Rendements log ---
df["rendement"] = np.log(df["close"] / df["close"].shift(1))

# --- 5. Volatilité ---
df["volatilite"] = df["rendement"].abs()

# --- 6. Supprimer NaN ---
df = df.dropna().reset_index(drop=True)

seuil = df["volatilite"].quantile(0.999)
df = df[df["volatilite"] < seuil].reset_index(drop=True)
print(f"Seuil aberrant retiré : {seuil:.4f}")

print(f"Nombre de jours pour {TICKER} : {len(df)}")
print(df[["date", "close", "rendement", "volatilite"]].head(10))

# --- 7. Graphiques ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(df["date"], df["close"], color="#1D9E75", linewidth=0.8)
axes[0].set_title(f"Prix de clôture — {TICKER}")
axes[0].set_ylabel("Prix ($)")

axes[1].plot(df["date"], df["rendement"], color="#534AB7", linewidth=0.6)
axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
axes[1].set_title("Rendements logarithmiques")
axes[1].set_ylabel("Rendement")

axes[2].plot(df["date"], df["volatilite"], color="#D85A30", linewidth=0.6)
axes[2].set_title("Volatilité (|rendement|)")
axes[2].set_ylabel("Volatilité")

plt.tight_layout()
plt.savefig("figures/01_serie_nvda.png", dpi=150)
plt.show()
print("Graphique sauvegardé dans figures/")

# --- 8. Sauvegarde ---
df.to_csv("data/nvda_clean.csv", index=False)
print("Fichier sauvegardé : data/nvda_clean.csv")