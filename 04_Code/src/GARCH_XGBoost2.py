import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import math

# 1) Daten einlesen und Renditen berechnen
ticker = "NVDA"
df = pd.read_csv(
    f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
    parse_dates=["Date"], index_col="Date"
)
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) TimeSeriesSplit konfigurieren
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Fenstergröße
window_size = 10

def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        feats = list(rets[i-window_size:i]) + [vols[i]]
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

# Metrik-Listen
metrics_returns = []
metrics_prices = []

last_fold = None

fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 4) GARCH fitten
    scaled = train_df["Return"] * 10
    g = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = g.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_vol = np.sqrt(fc.variance.values[-1, :]) / 10
    test_df["GARCH_vol"] = test_vol

    # 6) Features und Targets
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    # 7) XGBoost trainieren
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 8) Metriken Log-Renditen
    rmse_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_ret = mean_absolute_error(y_test, y_pred)
    mape_ret = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2_ret = r2_score(y_test, y_pred)
    hitrate_ret = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
    sharpe_ret = np.mean(y_pred) / (np.std(y_pred) + 1e-8)

    metrics_returns.append([rmse_ret, mae_ret, mape_ret, r2_ret, hitrate_ret, sharpe_ret])

    # 9) Rückrechnung auf Preisbasis
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx = i + window_size
        prev_p = test_df["Close"].iloc[idx-1]
        p_pred = prev_p * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])

    preds = np.array(preds)
    actuals = np.array(actuals)

    # 10) Metriken Preise
    rmse_price = math.sqrt(mean_squared_error(actuals, preds))
    mae_price = mean_absolute_error(actuals, preds)
    mape_price = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    r2_price = r2_score(actuals, preds)
    hitrate_price = np.mean(np.sign(np.diff(actuals)) == np.sign(np.diff(preds))) * 100
    sharpe_price = np.mean(np.diff(preds)) / (np.std(np.diff(preds)) + 1e-8)

    metrics_prices.append([rmse_price, mae_price, mape_price, r2_price, hitrate_price, sharpe_price])

    print(f"Fold {fold}: RMSE_Returns={rmse_ret:.4f}, RMSE_Prices={rmse_price:.4f}")

    if fold == n_splits:
        last_fold = {
            "index": test_df.index[window_size:],
            "y_test": y_test,
            "y_pred": y_pred,
            "actuals": actuals,
            "preds": preds
        }
    fold += 1

# 11) Durchschnittliche Metriken ausgeben
def print_avg_metrics(name, metrics):
    m = np.array(metrics)
    print(f"\n=== Durchschnittliche Metriken: {name} ===")
    print(f"RMSE  = {np.mean(m[:,0]):.4f}")
    print(f"MAE   = {np.mean(m[:,1]):.4f}")
    print(f"MAPE  = {np.mean(m[:,2]):.2f}%")
    print(f"R²    = {np.mean(m[:,3]):.4f}")
    print(f"HitRate = {np.mean(m[:,4]):.2f}%")
    print(f"Sharpe = {np.mean(m[:,5]):.4f}")

print_avg_metrics("Log-Renditen", metrics_returns)
print_avg_metrics("Preise", metrics_prices)

# 12) Plot letzter Fold
if last_fold:
    idx = last_fold["index"]

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_test"],  label="Tatsächliche Log-Renditen")
    plt.plot(idx, last_fold["y_pred"],  label="Predicted Log-Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log-Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["actuals"], label="Tatsächlicher Kurs")
    plt.plot(idx, last_fold["preds"],   label="Predicted Kurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Price (Close)"); plt.legend()
    plt.show()
