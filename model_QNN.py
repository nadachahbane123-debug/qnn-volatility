# ============================================================
# FICHIER : model_QNN.py
# RÔLE    : Quantile Neural Network avec PyTorch
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Reproductibilité — pour avoir les mêmes résultats à chaque run
torch.manual_seed(42)
np.random.seed(42)

print("PyTorch version :", torch.__version__)

# --- 1. Charger les données ---
X_train = np.load("data/X_train.npy").astype(np.float32)
X_test  = np.load("data/X_test.npy").astype(np.float32)
y_train = np.load("data/y_train.npy").astype(np.float32)
y_test  = np.load("data/y_test.npy").astype(np.float32)
dates_test = np.load("data/dates_test.npy", allow_pickle=True)

print(f"Train : {X_train.shape}  —  Test : {X_test.shape}")

# --- 2. Convertir en tenseurs PyTorch ---
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# DataLoader pour entraîner par mini-batchs
dataset    = TensorDataset(X_train_t, y_train_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

print("Tenseurs créés")

# --- 3. Architecture du réseau ---
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

# --- 4. Pinball loss ---
def pinball_loss(y_pred, y_true, tau):
    erreur = y_true - y_pred
    loss = torch.where(erreur >= 0,
                       tau * erreur,
                       (tau - 1) * erreur)
    return loss.mean()

# --- 5. Entraînement pour les 3 quantiles ---
quantiles   = [0.1, 0.5, 0.9]
modeles_qnn = {}
historiques = {}

EPOCHS      = 100
LR          = 0.001

for tau in quantiles:
    print(f"\nEntraînement QNN τ={tau}...")
    
    modele    = QNN(n_features=8)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)
    historique = []

    for epoch in range(EPOCHS):
        modele.train()
        perte_epoch = 0

        for X_batch, y_batch in dataloader:
            optimiseur.zero_grad()
            y_pred = modele(X_batch)
            perte  = pinball_loss(y_pred, y_batch, tau)
            perte.backward()
            optimiseur.step()
            perte_epoch += perte.item()

        perte_moy = perte_epoch / len(dataloader)
        historique.append(perte_moy)

        if (epoch + 1) % 20 == 0:
            print(f"  Époque {epoch+1}/{EPOCHS} — perte : {perte_moy:.6f}")

    modeles_qnn[tau]  = modele
    historiques[tau]  = historique
    print(f"  τ={tau} — TERMINE")

# --- 6. Courbes de perte ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
couleurs  = {0.1: "#1D9E75", 0.5: "#534AB7", 0.9: "#D85A30"}

for i, tau in enumerate(quantiles):
    axes[i].plot(historiques[tau], color=couleurs[tau], linewidth=1.2)
    axes[i].set_title(f"Perte entraînement — τ={tau}")
    axes[i].set_xlabel("Époque")
    axes[i].set_ylabel("Pinball loss")

plt.tight_layout()
plt.savefig("figures/06_courbes_perte_QNN.png", dpi=150)
plt.show()
print("Courbes de perte sauvegardées")


# --- 7. Prédictions ---
predictions_qnn = {}

for tau in quantiles:
    modeles_qnn[tau].eval()
    with torch.no_grad():
        y_pred = modeles_qnn[tau](X_test_t).numpy().flatten()
    predictions_qnn[tau] = y_pred

print("\nPrédictions calculées")


# --- 8. Métriques ---
def pinball_np(y_true, y_pred, tau):
    erreur = y_true - y_pred
    return np.mean(np.where(erreur >= 0, tau*erreur, (tau-1)*erreur))

def mape(y_true, y_pred):
    masque = y_true > 0.001
    return np.mean(np.abs((y_true[masque]-y_pred[masque])/y_true[masque]))*100

print("\n--- Métriques QNN ---")
print(f"{'τ':<6} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Pinball':<10}")
print("-" * 46)

resultats_qnn = {}
for tau in quantiles:
    y_pred  = predictions_qnn[tau]
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae     = mean_absolute_error(y_test, y_pred)
    mape_v  = mape(y_test, y_pred)
    pinball = pinball_np(y_test, y_pred, tau)
    resultats_qnn[tau] = {"RMSE": rmse, "MAE": mae,
                           "MAPE": mape_v, "Pinball": pinball}
    print(f"{tau:<6} {rmse:<10.4f} {mae:<10.4f} {mape_v:<10.1f} {pinball:<10.4f}")


# --- 9. Graphique prédictions ---
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

for i, tau in enumerate(quantiles):
    axes[i].plot(dates_test, y_test,
                 color="gray", linewidth=0.5,
                 alpha=0.7, label="Volatilité réelle")
    axes[i].plot(dates_test, predictions_qnn[tau],
                 color=couleurs[tau], linewidth=1.2,
                 label=f"QNN τ={tau}")
    axes[i].set_ylabel("Volatilité")
    axes[i].legend(loc="upper right")
    axes[i].set_title(f"QNN — τ = {tau}")

plt.suptitle("Prédictions QNN vs Volatilité réelle — NVDA",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/07_predictions_QNN.png", dpi=150)
plt.show()
print("Graphique prédictions sauvegardé")

# --- 10. Graphique comparaison QR vs QNN ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, tau in enumerate(quantiles):
    qr_pred  = pd.read_csv("data/predictions_QR.csv")[f"QR_tau0{int(tau*10)}"].values
    qnn_pred = predictions_qnn[tau]

    axes[i].plot(dates_test, y_test,
                 color="gray", linewidth=0.5, alpha=0.6, label="Réel")
    axes[i].plot(dates_test, qr_pred,
                 color="#534AB7", linewidth=1, 
                 alpha=0.8, label="QR")
    axes[i].plot(dates_test, qnn_pred,
                 color="#D85A30", linewidth=1,
                 alpha=0.8, label="QNN")
    axes[i].set_title(f"τ = {tau}")
    axes[i].legend()
    axes[i].set_ylabel("Volatilité")

plt.suptitle("Comparaison QR vs QNN — NVDA", fontsize=13)
plt.tight_layout()
plt.savefig("figures/08_comparaison_QR_QNN.png", dpi=150)
plt.show()
print("Graphique comparaison sauvegardé")


# --- 11. Sauvegarder prédictions et modèles ---
df_pred = pd.DataFrame({
    "date"      : dates_test,
    "y_reel"    : y_test,
    "QNN_tau01" : predictions_qnn[0.1],
    "QNN_tau05" : predictions_qnn[0.5],
    "QNN_tau09" : predictions_qnn[0.9],
})
df_pred.to_csv("data/predictions_QNN.csv", index=False)

for tau in quantiles:
    torch.save(modeles_qnn[tau].state_dict(),
               f"data/qnn_tau{int(tau*10)}.pt")

print("\nPrédictions sauvegardées : data/predictions_QNN.csv")
print("Modèles sauvegardés : data/qnn_tau*.pt")
print("\nModèle QNN — TERMINE !")


