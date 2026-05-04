# ============================================================
# FICHIER : pipeline.py
# RÔLE    : fonctions réutilisables pour n'importe quel ticker
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf
import shap
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag5",
            "vol_moy_5j", "vol_moy_10j", "vol_moy_20j"]
QUANTILES = [0.1, 0.5, 0.9]

# ── Architecture QNN (identique à model_QNN.py) ──
class QNN(nn.Module):
    def __init__(self, n_features=8):
        super(QNN, self).__init__()
        self.reseau = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.reseau(x)

def pinball_loss_torch(y_pred, y_true, tau):
    e = y_true - y_pred
    return torch.where(e >= 0, tau * e, (tau - 1) * e).mean()

def pinball_loss_np(y_true, y_pred, tau):
    e = y_true - y_pred
    return np.mean(np.where(e >= 0, tau * e, (tau - 1) * e))

def mape_score(y_true, y_pred):
    mask = y_true > 0.001
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ── Étape 1 : charger et préparer les données ──
def preparer_donnees(ticker, csv_path="nasdaq100_merged_1.csv"):
    df = pd.read_csv(csv_path)
    df = df[df["ticker"] == ticker].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["rendement"]  = np.log(df["close"] / df["close"].shift(1))
    df["volatilite"] = df["rendement"].abs()
    df = df.dropna().reset_index(drop=True)
    # Supprimer aberrations
    seuil = df["volatilite"].quantile(0.999)
    df = df[df["volatilite"] < seuil].reset_index(drop=True)
    return df

# ── Étape 2 : créer les features ──
def creer_features(df):
    for i in range(1, 6):
        df[f"lag{i}"] = df["volatilite"].shift(i)
    df["vol_moy_5j"]  = df["volatilite"].shift(1).rolling(5).mean()
    df["vol_moy_10j"] = df["volatilite"].shift(1).rolling(10).mean()
    df["vol_moy_20j"] = df["volatilite"].shift(1).rolling(20).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# ── Étape 3 : split + normalisation ──
def splitter(df):
    X = df[FEATURES].values
    y = df["volatilite"].values
    dates = df["date"].values
    split = int(len(df) * 0.80)
    X_train, X_test   = X[:split], X[split:]
    y_train, y_test   = y[:split], y[split:]
    dates_test        = dates[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, dates_test, scaler

# ── Étape 4 : entraîner QR ──
def entrainer_QR(X_train, y_train, X_test):
    df_train = pd.DataFrame(X_train, columns=FEATURES)
    df_train["volatilite"] = y_train
    df_test  = pd.DataFrame(X_test,  columns=FEATURES)
    formule  = "volatilite ~ " + " + ".join(FEATURES)
    preds = {}
    for tau in QUANTILES:
        modele       = smf.quantreg(formule, df_train).fit(q=tau, max_iter=2000)
        preds[tau]   = modele.predict(df_test).values
    return preds

# ── Étape 5 : entraîner QNN ──
def entrainer_QNN(X_train, y_train, X_test, epochs=100, lr=0.001):
    from torch.utils.data import DataLoader, TensorDataset
    X_tr = torch.FloatTensor(X_train.astype(np.float32))
    y_tr = torch.FloatTensor(y_train.astype(np.float32)).unsqueeze(1)
    X_te = torch.FloatTensor(X_test.astype(np.float32))
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=False)
    preds  = {}
    for tau in QUANTILES:
        modele = QNN()
        optim  = torch.optim.Adam(modele.parameters(), lr=lr)
        for _ in range(epochs):
            modele.train()
            for xb, yb in loader:
                optim.zero_grad()
                pinball_loss_torch(modele(xb), yb, tau).backward()
                optim.step()
        modele.eval()
        with torch.no_grad():
            preds[tau] = modele(X_te).numpy().flatten()
    return preds, modele

# ── Étape 6 : calculer les métriques ──
def calculer_metriques(y_test, preds_qr, preds_qnn):
    rows = []
    for tau in QUANTILES:
        for nom, preds in [("QR", preds_qr), ("QNN", preds_qnn)]:
            yp = preds[tau]
            rows.append({
                "Modèle"  : nom,
                "τ"       : tau,
                "RMSE"    : round(np.sqrt(mean_squared_error(y_test, yp)), 4),
                "MAE"     : round(mean_absolute_error(y_test, yp), 4),
                "MAPE"    : round(mape_score(y_test, yp), 1),
                "Pinball" : round(pinball_loss_np(y_test, yp, tau), 4),
            })
    return pd.DataFrame(rows)

# ── Étape 7 : calculer SHAP ──
def calculer_shap(X_train, X_test):
    modele = QNN()
    X_tr   = torch.FloatTensor(X_train[:200].astype(np.float32))
    X_te   = torch.FloatTensor(X_test.astype(np.float32))
    explainer   = shap.DeepExplainer(modele, X_tr)
    shap_values = explainer.shap_values(X_te)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    return shap_values

# ── Fonction principale : tout en un ──
def run_pipeline(ticker):
    df           = preparer_donnees(ticker)
    df           = creer_features(df)
    X_tr, X_te, y_tr, y_te, dates, scaler = splitter(df)
    preds_qr     = entrainer_QR(X_tr, y_tr, X_te)
    preds_qnn, _ = entrainer_QNN(X_tr, y_tr, X_te)
    metriques    = calculer_metriques(y_te, preds_qr, preds_qnn)
    shap_vals    = calculer_shap(X_tr, X_te)
    return df, y_te, dates, preds_qr, preds_qnn, metriques, shap_vals

