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
window_size = 10

# 3) Helper: Features und Target aus Returns + GARCH-Vol erstellen
def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        feats = np.concatenate([rets[i-window_size:i], [vols[i]]])
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

# 4) Speicher für Metriken und Plot-Daten
metrics_returns = []
metrics_prices  = []

dates_ret_all = []
y_test_all    = []
y_pred_all    = []

dates_pr_all  = []
actuals_all   = []
preds_all     = []

fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # a) GARCH auf Trainingsdaten fitten
    scaled = train_df["Return"] * 10
    g = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                   dist="normal", rescale=False)
    res = g.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # b) Forecast der Volatilität für Testperiode
    fc = res.forecast(start=train_df.index[-1],
                      horizon=len(test_df),
                      reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # c) Features & Targets für XGB
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    # d) XGBoost trainieren und vorhersagen
    model = XGBRegressor(n_estimators=100, max_depth=3,
                         learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Datums-Indizes
    dates = test_df.index[window_size:]
    # für Renditen-Plot
    dates_ret_all.append(dates)
    y_test_all.append(y_test)
    y_pred_all.append(y_pred)

    # e) Metriken für Log-Renditen
    rmse_ret    = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_ret     = mean_absolute_error(y_test, y_pred)
    mape_ret    = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2_ret      = r2_score(y_test, y_pred)
    hitrate_ret = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
    sharpe_ret  = np.mean(y_pred) / (np.std(y_pred) + 1e-8)

    metrics_returns.append([
        rmse_ret, mae_ret, mape_ret,
        r2_ret, hitrate_ret, sharpe_ret
    ])

    # f) Rückrechnung in Close-Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx = i + window_size
        prev_p = test_df["Close"].iloc[idx-1]
        p_pred = prev_p * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])
    preds   = np.array(preds)
    actuals = np.array(actuals)

    # für Preis-Plot
    dates_pr_all.append(dates)
    actuals_all.append(actuals)
    preds_all.append(preds)

    # g) Metriken auf Preisbasis
    rmse_price   = math.sqrt(mean_squared_error(actuals, preds))
    mae_price    = mean_absolute_error(actuals, preds)
    mape_price   = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    r2_price     = r2_score(actuals, preds)
    hitrate_price= np.mean(np.sign(np.diff(actuals)) ==
                           np.sign(np.diff(preds))) * 100
    sharpe_price = np.mean(np.diff(preds)) / (np.std(np.diff(preds)) + 1e-8)

    metrics_prices.append([
        rmse_price, mae_price, mape_price,
        r2_price, hitrate_price, sharpe_price
    ])

    print(f"Fold {fold}: RMSE_Returns={rmse_ret:.4f}, RMSE_Prices={rmse_price:.4f}")
    fold += 1

# 5) Durchschnittliche Metriken ausgeben
def print_avg(name, arr):
    m = np.array(arr)
    print(f"\n=== Ø Metriken: {name} ===")
    print(f"RMSE   = {m[:,0].mean():.4f}")
    print(f"MAE    = {m[:,1].mean():.4f}")
    print(f"MAPE   = {m[:,2].mean():.2f}%")
    print(f"R²     = {m[:,3].mean():.4f}")
    print(f"HitRate= {m[:,4].mean():.2f}%")
    print(f"Sharpe = {m[:,5].mean():.4f}")

print_avg("Log-Renditen", metrics_returns)
print_avg("Close-Preise", metrics_prices)

# 6) Plot aller 3 Folds: Log-Renditen
plt.figure(figsize=(12, 6))
for i in range(n_splits):
    plt.plot(dates_ret_all[i],    y_test_all[i],
             label=f"Real Ret Fold {i+1}")
    plt.plot(dates_ret_all[i],    y_pred_all[i],
             linestyle="--", label=f"Pred Ret Fold {i+1}")
plt.title(f"{ticker} – Log-Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log-Return")
plt.legend(); plt.tight_layout(); plt.show()

# 7) Plot aller 3 Folds: Close-Preise
plt.figure(figsize=(12, 6))
for i in range(n_splits):
    plt.plot(dates_pr_all[i], actuals_all[i],
             label=f"Real Prc Fold {i+1}")
    plt.plot(dates_pr_all[i], preds_all[i],
             linestyle="--", label=f"Pred Prc Fold {i+1}")
plt.title(f"{ticker} – Close-Preise alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)")
plt.legend(); plt.tight_layout(); plt.show()
