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
csv_path     = f"../../03_Daten/processed_data/merged_weekly_{ticker}.csv"

# 1) Daten einlesen und Log‑Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Definiere die Google‑Trends‑Spalten und stelle sicher, dass RSI_14 bereits in df ist
gtd_cols = [c for c in df.columns if "stock" in c.lower()]
static_cols = ["GARCH_vol", "RSI_14"] + gtd_cols

# 3) Cross‑Validation
tscv = TimeSeriesSplit(n_splits=n_splits)

# 4) Feature‑Creator
def create_features_and_target(subdf, window_size=10):
    X, y = [], []
    rets = subdf["Return"].values
    for i in range(window_size, len(subdf)):
        # a) Sequenz der letzten window_size Log‑Renditen
        seq = list(rets[i-window_size:i])
        # b) statische Features zum Zeitpunkt i
        stat = [subdf[col].iat[i] for col in static_cols]
        X.append(seq + stat)
        y.append(rets[i])
    return np.array(X), np.array(y)

# 5) Listen für Ergebnisse
rmse_returns = []
rmse_prices  = []
last_fold    = {}

# 6) Schleife über die Folds
for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # GARCH auf Trainings‑Returns fitten
    scaled = train_df["Return"] * 10
    garch = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    res   = garch.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # GARCH‑Forecast für Testbereich
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Features und Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)
    print(f"Fold {fold}: X_train {X_train.shape}, X_test {X_test.shape}")

    # XGBoost trainieren
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    # RMSE auf Log‑Renditen
    rm_r = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_r)

    # Rückrechnung der Renditen in Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx    = i + window_size
        p_prev = test_df["Close"].iloc[idx-1]
        preds.append(p_prev * np.exp(r))
        actuals.append(test_df["Close"].iloc[idx])
    rm_p = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_p)

    print(f"Fold {fold}: RMSE_Returns={rm_r:.4f}, RMSE_Prices={rm_p:.4f}")

    # Daten für letzten Fold merken
    if fold == n_splits:
        last_fold = {
            "idx":      test_df.index[window_size:],
            "y_test":   y_test,
            "y_pred":   y_pred,
            "actuals":  actuals,
            "preds":    preds
        }

# 7) Durchschnittliche RMSE ausgeben
print(f"\nDurchschn. RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschn. RMSE Preise:       {np.mean(rmse_prices):.4f}")

# 8) Plots für den letzten Fold
if last_fold:
    idx = last_fold["idx"]

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_test"],  label="Actual Log‑Renditen")
    plt.plot(idx, last_fold["y_pred"],  label="Predicted Log‑Renditen", alpha=0.7)
    plt.title(f"{ticker} – Fold {n_splits}: Log‑Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log‑Rendite"); plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["actuals"], label="Actual Close")
    plt.plot(idx, last_fold["preds"],   label="Predicted Close", alpha=0.7)
    plt.title(f"{ticker} – Fold {n_splits}: Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend()
    plt.show()
