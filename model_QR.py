# ============================================================
# FICHIER : model_QR.py
# RÔLE    : régression quantile linéaire pour τ = 0.1, 0.5, 0.9
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- 1. Charger les données ---
X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
dates_test = np.load("data/dates_test.npy", allow_pickle=True)

print(f"Train : {X_train.shape}  —  Test : {X_test.shape}")

# --- 2. Préparer les données pour statsmodels ---
FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag5",
            "vol_moy_5j", "vol_moy_10j", "vol_moy_20j"]

# Créer des dataframes avec les bons noms de colonnes
df_train = pd.DataFrame(X_train, columns=FEATURES)
df_train["volatilite"] = y_train

df_test = pd.DataFrame(X_test, columns=FEATURES)

# Formule du modèle
formule = "volatilite ~ " + " + ".join(FEATURES)
print(f"Formule : {formule}")

# --- 3. Entraîner les 3 modèles quantiles ---
quantiles = [0.1, 0.5, 0.9]
modeles   = {}
predictions = {}

for tau in quantiles:
    print(f"Entraînement QR τ={tau}...", end=" ")
    modele = smf.quantreg(formule, df_train).fit(q=tau, max_iter=2000)
    modeles[tau] = modele
    predictions[tau] = modele.predict(df_test)
    print("OK")

# --- 4. Calculer les métriques ---
def pinball_loss(y_true, y_pred, tau):
    erreur = y_true - y_pred
    return np.mean(np.where(erreur >= 0, tau * erreur, (tau - 1) * erreur))

def mape(y_true, y_pred):
    masque = y_true > 0.001
    return np.mean(np.abs((y_true[masque] - y_pred[masque]) / y_true[masque])) * 100

print("\n--- Métriques QR ---")
print(f"{'τ':<6} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Pinball':<10}")
print("-" * 46)

resultats_qr = {}
for tau in quantiles:
    y_pred = predictions[tau].values
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae     = mean_absolute_error(y_test, y_pred)
    mape_v  = mape(y_test, y_pred)
    pinball = pinball_loss(y_test, y_pred, tau)
    resultats_qr[tau] = {"RMSE": rmse, "MAE": mae, 
                          "MAPE": mape_v, "Pinball": pinball}
    print(f"{tau:<6} {rmse:<10.4f} {mae:<10.4f} {mape_v:<10.1f} {pinball:<10.4f}")


# --- 5. Graphique des prédictions ---
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
couleurs = {0.1: "#1D9E75", 0.5: "#534AB7", 0.9: "#D85A30"}

for i, tau in enumerate(quantiles):
    axes[i].plot(dates_test, y_test, 
                 color="gray", linewidth=0.5, 
                 alpha=0.7, label="Volatilité réelle")
    axes[i].plot(dates_test, predictions[tau].values,
                 color=couleurs[tau], linewidth=1.2,
                 label=f"QR τ={tau}")
    axes[i].set_ylabel("Volatilité")
    axes[i].legend(loc="upper right")
    axes[i].set_title(f"Régression quantile — τ = {tau}")

plt.suptitle("Prédictions QR vs Volatilité réelle — NVDA", 
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/04_predictions_QR.png", dpi=150)
plt.show()
print("Graphique sauvegardé dans figures/")

# --- 6. Graphique de couverture ---
# Un bon quantile τ=0.9 doit être dépassé ~10% du temps
print("\n--- Couverture empirique ---")
print(f"{'τ':<6} {'Attendu':<12} {'Observé':<12} {'Ecart':<10}")
print("-" * 40)

fig2, ax2 = plt.subplots(figsize=(7, 4))
taux_attendus  = []
taux_observes  = []

for tau in quantiles:
    y_pred   = predictions[tau].values
    couverture = np.mean(y_test <= y_pred) * 100
    attendu    = tau * 100
    ecart      = couverture - attendu
    taux_attendus.append(attendu)
    taux_observes.append(couverture)
    print(f"{tau:<6} {attendu:<12.1f} {couverture:<12.1f} {ecart:<+10.1f}")

ax2.plot(taux_attendus, taux_attendus, 
         "k--", linewidth=1, label="Calibration parfaite")
ax2.scatter(taux_attendus, taux_observes, 
            color=["#1D9E75","#534AB7","#D85A30"], s=100, zorder=5)
for i, tau in enumerate(quantiles):
    ax2.annotate(f"τ={tau}", (taux_attendus[i], taux_observes[i]),
                 textcoords="offset points", xytext=(8, 0), fontsize=10)
ax2.set_xlabel("Couverture attendue (%)")
ax2.set_ylabel("Couverture observée (%)")
ax2.set_title("Calibration des quantiles — QR")
ax2.legend()
plt.tight_layout()
plt.savefig("figures/05_calibration_QR.png", dpi=150)
plt.show()
print("Graphique calibration sauvegardé")

# --- 7. Sauvegarder les prédictions ---
df_pred = pd.DataFrame({
    "date"       : dates_test,
    "y_reel"     : y_test,
    "QR_tau01"   : predictions[0.1].values,
    "QR_tau05"   : predictions[0.5].values,
    "QR_tau09"   : predictions[0.9].values,
})
df_pred.to_csv("data/predictions_QR.csv", index=False)
print("\nPrédictions sauvegardées : data/predictions_QR.csv")
print("\nModèle QR — TERMINE !")

