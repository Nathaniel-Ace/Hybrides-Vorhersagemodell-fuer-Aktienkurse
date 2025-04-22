import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Einstellungen
ticker       = "NVDA"
window_size  = 10
n_splits     = 3
csv_path     = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Log‑Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) TimeSeriesSplit konfigurieren
tscv = TimeSeriesSplit(n_splits=n_splits)

# 3) Hilfsfunktion: Features und Ziel erstellen
def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    rsis = df["RSI_14"].values
    for i in range(window_size, len(df)):
        past_rets = list(rets[i-window_size:i])    # letzte window_size Log‑Renditen
        feat = past_rets + [vols[i], rsis[i]]       # + GARCH_vol + RSI_14 zum Zeitpunkt i
        X.append(feat)
        y.append(rets[i])                           # vorherzusagende Log‑Rendite
    return np.array(X), np.array(y)

# 4) Cross‑Validation und Training
rmse_returns = []
rmse_prices  = []
last_fold    = None

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 5) GARCH auf Trainingsdaten fitten (skaliert mit Faktor 10, rescale=False)
    scaled = train_df["Return"] * 10
    garch = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    res   = garch.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # 6) GARCH‑Volatilitäts‑Forecast für Testbereich
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # 7) Features und Targets für diesen Fold erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)
    print(f"Fold {fold}: X_train = {X_train.shape}, X_test = {X_test.shape}")

    # 8) XGBoost trainieren
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 9) RMSE auf Log‑Renditen
    rm_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_ret)

    # 10) Rückrechnung in Preis‑Basis
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx = i + window_size
        prev_p = test_df["Close"].iloc[idx-1]
        p_pred = prev_p * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])
    rm_price = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_price)
    print(f"Fold {fold}: RMSE_Returns={rm_ret:.4f}, RMSE_Prices={rm_price:.4f}")

    # 11) Letzten Fold für Plots merken
    if fold == n_splits:
        last_fold = {
            "idx":      test_df.index[window_size:],
            "y_test":   y_test,
            "y_pred":   y_pred,
            "actuals":  actuals,
            "preds":    preds
        }

# 12) Durchschnittliche RMSE ausgeben
print(f"\nDurchschnittlicher RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschnittlicher RMSE Aktienkurse: {np.mean(rmse_prices):.4f}")

# 13) Plots für den letzten Fold
if last_fold:
    idx = last_fold["idx"]

    plt.figure(figsize=(12, 5))
    plt.plot(idx, last_fold["y_test"],  label="Tatsächliche Log‑Renditen")
    plt.plot(idx, last_fold["y_pred"],  label="Vorhergesagte Log‑Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log‑Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log‑Rendite"); plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(idx, last_fold["actuals"], label="Tatsächlicher Aktienkurs")
    plt.plot(idx, last_fold["preds"],   label="Vorhergesagter Aktienkurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend()
    plt.show()
