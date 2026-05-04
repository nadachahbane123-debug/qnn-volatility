# ============================================================
# FICHIER : shap_analysis.py
# RÔLE    : Explainability avec SHAP sur le QNN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

print("Chargement des données et du modèle...")

# --- 1. Recharger les données ---
X_train = np.load("data/X_train.npy").astype(np.float32)
X_test  = np.load("data/X_test.npy").astype(np.float32)
y_test  = np.load("data/y_test.npy").astype(np.float32)
dates_test = np.load("data/dates_test.npy", allow_pickle=True)

FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag5",
            "vol_moy_5j", "vol_moy_10j", "vol_moy_20j"]

print(f"Données chargées : {X_test.shape}")

# --- 2. Reconstruire et recharger le QNN ---
class QNN(nn.Module):
    def __init__(self, n_features):
        super(QNN, self).__init__()
        self.reseau = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.reseau(x)

# On charge le modèle τ=0.9 — le plus intéressant pour l'XAI
modele = QNN(n_features=8)
modele.load_state_dict(torch.load("data/qnn_tau9.pt"))
modele.eval()
print("Modèle QNN τ=0.9 chargé")

# --- 3. Préparer les données pour SHAP ---
# SHAP a besoin d'un échantillon de "background" 
# (représentatif de la distribution des données)
# On prend 200 points du train comme background
background = torch.FloatTensor(X_train[:200])
X_test_t   = torch.FloatTensor(X_test)

print("Calcul des valeurs SHAP...")

# --- 4. Calculer les valeurs SHAP ---
explainer   = shap.DeepExplainer(modele, background)
shap_values = explainer.shap_values(X_test_t)

# Aplatir la dimension supplémentaire (1342, 8, 1) -> (1342, 8)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

print(f"Valeurs SHAP calculées : {shap_values.shape}")

# --- 5. Graphique 1 : importance globale (bar plot) ---
shap_importance = np.abs(shap_values).mean(axis=0)
indices_tries   = np.argsort(shap_importance)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
couleurs_bar = ["#534AB7" if i == 0 else "#AFA9EC" 
                for i in range(len(FEATURES))]

bars = ax.barh(
    [FEATURES[i] for i in indices_tries[::-1]],
    shap_importance[indices_tries[::-1]],
    color=couleurs_bar[::-1]
)
ax.set_xlabel("Importance SHAP moyenne (|valeur|)")
ax.set_title("Importance globale des variables — QNN τ=0.9")
ax.axvline(x=0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("figures/09_shap_importance.png", dpi=150)
plt.show()
print("Graphique importance sauvegardé")

# --- 6. Graphique 2 : beeswarm plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES,
    show=False,
    plot_size=None
)
plt.title("SHAP Summary Plot — QNN τ=0.9")
plt.tight_layout()
plt.savefig("figures/10_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("SHAP summary plot sauvegardé")

# --- 7. Graphique 3 : analyse locale sur 3 jours clés ---
# Trouver les jours les plus intéressants
idx_forte  = np.argmax(y_test)           # jour de volatilité max
idx_calme  = np.argmin(y_test)           # jour le plus calme
idx_median = np.argsort(y_test)[len(y_test)//2]  # jour médian

jours_cles = {
    f"Forte volatilité\n({str(dates_test[idx_forte])[:10]})": idx_forte,
    f"Volatilité médiane\n({str(dates_test[idx_median])[:10]})": idx_median,
    f"Faible volatilité\n({str(dates_test[idx_calme])[:10]})": idx_calme,
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (titre, idx) in zip(axes, jours_cles.items()):
    vals  = shap_values[idx]
    ordre = np.argsort(np.abs(vals))[::-1]
    
    couleurs = ["#D85A30" if v > 0 else "#534AB7" for v in vals[ordre]]
    
    axes[list(jours_cles.keys()).index(titre)].barh(
        [FEATURES[i] for i in ordre[::-1]],
        vals[ordre[::-1]],
        color=couleurs[::-1]
    )
    axes[list(jours_cles.keys()).index(titre)].set_title(titre)
    axes[list(jours_cles.keys()).index(titre)].axvline(
        x=0, color="black", linewidth=0.8
    )
    axes[list(jours_cles.keys()).index(titre)].set_xlabel("Contribution SHAP")

plt.suptitle("Analyse locale SHAP — 3 jours clés", fontsize=13)
plt.tight_layout()
plt.savefig("figures/11_shap_local.png", dpi=150)
plt.show()
print("Analyse locale sauvegardée")

# --- 8. Résumé textuel ---
print("\n--- Importance globale des variables ---")
print(f"{'Variable':<15} {'Importance SHAP':<20}")
print("-" * 35)
for i in indices_tries:
    print(f"{FEATURES[i]:<15} {shap_importance[i]:<20.6f}")

print("\n--- Interprétation ---")
var_top1 = FEATURES[indices_tries[0]]
var_top2 = FEATURES[indices_tries[1]]
var_top3 = FEATURES[indices_tries[2]]
print(f"Variable la plus importante  : {var_top1}")
print(f"Variable 2ème plus importante: {var_top2}")
print(f"Variable 3ème plus importante: {var_top3}")

# Sauvegarder les valeurs SHAP
df_shap = pd.DataFrame(shap_values, columns=FEATURES)
df_shap.to_csv("data/shap_values.csv", index=False)
print("\nValeurs SHAP sauvegardées : data/shap_values.csv")
print("\nAnalyse SHAP — TERMINEE !")

