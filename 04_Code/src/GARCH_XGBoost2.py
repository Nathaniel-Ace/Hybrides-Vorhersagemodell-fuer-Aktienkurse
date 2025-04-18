import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import math

# 1) Daten einlesen und Renditen berechnen
df = pd.read_csv(
    "../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv",
    parse_dates=["Date"], index_col="Date"
)
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) TimeSeriesSplit konfigurieren
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Fenstergröße für Features
window_size = 10

# Hilfsfunktion: X, y aus Renditen + GARCH_vol erstellen
def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        feats = list(rets[i-window_size:i]) + [vols[i]]
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

# Listen, um RMSE pro Fold zu speichern
rmse_returns = []
rmse_prices = []

# letzten Fold fürs Plotten merken
last_fold = None

# 3) Cross‑Validation über Folds
fold = 1
for train_idx, test_idx in tscv.split(df):
    # Split in DataFrames
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 4) GARCH auf Trainingsdaten fitten
    scaled = train_df["Return"] * 10
    g = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = g.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # 5) Volatilitäts‑Forecast für den Testbereich
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_vol = np.sqrt(fc.variance.values[-1, :]) / 10
    test_df["GARCH_vol"] = test_vol

    # 6) Features und Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    # 7) XGBoost auf log‑Renditen trainieren
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 8) RMSE auf Log‑Renditen
    rm_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_ret)

    # 9) Rückrechnung in Preisbasis
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

    # Merke letzten Fold fürs Plot
    if fold == n_splits:
        last_fold = {
            "index": test_df.index[window_size:],
            "y_test": y_test,
            "y_pred": y_pred,
            "actuals": actuals,
            "preds": preds
        }
    fold += 1

# 10) Durchschnittliche RMSE ausgeben
print(f"\nDurchschnittlicher RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschnittlicher RMSE Aktienkurse: {np.mean(rmse_prices):.4f}")

# 11) Plots für den letzten Fold
if last_fold:
    idx = last_fold["index"]

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_test"],  label="Tatsächliche Log‑Renditen")
    plt.plot(idx, last_fold["y_pred"],  label="Predicted Log‑Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log‑Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log‑Rendite"); plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["actuals"], label="Tatsächlicher Kurs")
    plt.plot(idx, last_fold["preds"],   label="Predicted Kurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Price (Close)"); plt.legend()
    plt.show()
