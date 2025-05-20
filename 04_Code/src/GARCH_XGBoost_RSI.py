import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Einstellungen
ticker      = "MSFT"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Log-Renditen berechnen
df = (
    pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
      .sort_index()
)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) GARCH einmalig auf gesamten Datensatz fitten und Volatilität speichern
res_all = arch_model(df["Return"] * 10,
                     mean="Zero", vol="GARCH", p=1, q=1,
                     dist="normal", rescale=False
                    ).fit(disp="off")
df["GARCH_vol"] = res_all.conditional_volatility / 10

# 3) Helper zum Erzeugen der Features/Targets
def make_xy(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    rsis = df["RSI_14"].values
    for i in range(window_size, len(df)):
        seq   = rets[i-window_size:i]
        feats = np.concatenate([seq, [vols[i], rsis[i]]])
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

# 4) Hyperparam-Suche
param_dist = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "gamma":           [0, 0.1, 0.5]
}
xgb_base = XGBRegressor(random_state=42, tree_method="hist")
tscv_search = TimeSeriesSplit(n_splits=n_splits)
search = RandomizedSearchCV(
    xgb_base, param_dist,
    n_iter=20,
    cv=tscv_search,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=0
)
X_full, y_full = make_xy(df, window_size)
search.fit(X_full, y_full)
best_params = search.best_params_
print(">>> Best XGBoost Params:", best_params)

# 5) CV: Train/Test, Metriken, und Sammeln für Plots
tscv = TimeSeriesSplit(n_splits=n_splits)

# Listen für durchschnittliche Metriken
metrics_returns = []
metrics_prices  = []

# Listen für Plot-Daten aller Folds
dates_ret_all    = []
ret_true_all     = []
ret_pred_all     = []
dates_price_all  = []
price_true_all   = []
price_pred_all   = []

fold = 1
for tr_idx, te_idx in tscv.split(df):
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # GARCH-Forecast für Test-Set
    fc = res_all.forecast(
        start=train_df.index[-1],
        horizon=len(test_df),
        reindex=False
    )
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Features/Targets
    X_train, y_train = make_xy(train_df, window_size)
    X_test,  y_test  = make_xy(test_df,  window_size)

    # Finales Modell trainieren
    model = XGBRegressor(**best_params, random_state=42, tree_method="hist")
    model.fit(X_train, y_train)

    # Vorhersage Log-Returns
    y_pred = model.predict(X_test)

    # Datums-Indizes
    dates = test_df.index[window_size:]
    dates_ret_all.append(dates)
    ret_true_all.append(y_test)
    ret_pred_all.append(y_pred)

    # Metriken Log-Returns mit Filter für Null-Returns
    rmse_ret    = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_ret     = mean_absolute_error(y_test, y_pred)
    mask        = (y_test != 0)
    mape_ret    = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    r2_ret      = r2_score(y_test, y_pred)
    hitrate_ret = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
    sharpe_ret  = np.mean(y_pred) / (np.std(y_pred) + 1e-8)
    metrics_returns.append([rmse_ret, mae_ret, mape_ret,
                             r2_ret, hitrate_ret, sharpe_ret])

    # Rückrechnung auf Close-Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iat[i + window_size - 1]
        p_pred = p0 * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iat[i + window_size])
    preds   = np.array(preds)
    actuals = np.array(actuals)

    dates_price_all.append(dates)
    price_true_all.append(actuals)
    price_pred_all.append(preds)

    # Metriken Close-Preise
    rmse_p    = math.sqrt(mean_squared_error(actuals, preds))
    mae_p     = mean_absolute_error(actuals, preds)
    mape_p    = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    r2_p      = r2_score(actuals, preds)
    hitrate_p = np.mean(np.sign(np.diff(actuals)) == np.sign(np.diff(preds))) * 100
    sharpe_p  = np.mean(np.diff(preds)) / (np.std(np.diff(preds)) + 1e-8)
    metrics_prices.append([rmse_p, mae_p, mape_p,
                           r2_p, hitrate_p, sharpe_p])

    print(f"Fold {fold}: RMSE_Returns={rmse_ret:.4f}, RMSE_Prices={rmse_p:.4f}")
    fold += 1

# 6) Durchschnittliche Metriken ausgeben
def print_avg(name, arr):
    m = np.array(arr)
    print(f"\n=== Ø Metriken: {name} ===")
    print(f"RMSE    = {m[:,0].mean():.4f}")
    print(f"MAE     = {m[:,1].mean():.4f}")
    print(f"MAPE    = {m[:,2].mean():.2f}%")
    print(f"R²      = {m[:,3].mean():.4f}")
    print(f"HitRate = {m[:,4].mean():.2f}%")
    print(f"Sharpe  = {m[:,5].mean():.4f}")

print_avg("Log-Renditen", metrics_returns)
print_avg("Close-Preise", metrics_prices)

# 7) Plot: Log-Renditen über alle 3 Folds
plt.figure(figsize=(12,6))
for i in range(n_splits):
    plt.plot(dates_ret_all[i], ret_true_all[i],  label=f"Real Ret Fold {i+1}")
    plt.plot(dates_ret_all[i], ret_pred_all[i], linestyle="--", label=f"Pred Ret Fold {i+1}")
plt.title(f"{ticker} – Log-Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log-Return")
plt.legend(); plt.tight_layout(); plt.show()

# 8) Plot: Close-Preise über alle 3 Folds
plt.figure(figsize=(12,6))
for i in range(n_splits):
    plt.plot(dates_price_all[i], price_true_all[i],  label=f"Real Prc Fold {i+1}")
    plt.plot(dates_price_all[i], price_pred_all[i], linestyle="--", label=f"Pred Prc Fold {i+1}")
plt.title(f"{ticker} – Close-Preise alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)")
plt.legend(); plt.tight_layout(); plt.show()

from joblib import dump

# 9) Finales Modell auf allen historischen Daten trainieren
model_final = XGBRegressor(**best_params, random_state=42, tree_method="hist")
model_final.fit(X_full, y_full)

# 10) Vorhersage auf Trainingsdaten
y_full_pred = model_final.predict(X_full)

# RMSE Log-Rendite (Training)
rmse_ret_full = math.sqrt(mean_squared_error(y_full, y_full_pred))
print(f"\n📈 RMSE (Renditeebene) auf Trainingsdaten: {rmse_ret_full:.4f}")

# Preis-Rückrechnung
start_prices = df["Close"].iloc[window_size - 1 : -1].values
true_prices  = df["Close"].iloc[window_size:].values
pred_prices  = start_prices * np.exp(y_full_pred)

# RMSE Preis (Training)
rmse_prc_full = math.sqrt(mean_squared_error(true_prices, pred_prices))
print(f"💰 RMSE (Preisebene) auf Trainingsdaten:  {rmse_prc_full:.4f}")

# 11) Modell speichern
model_path = f"../../05_Modelle/garch_xgboost_{ticker.lower()}_final_model.joblib"
dump(model_final, model_path)
print(f"✅ Modell gespeichert: {model_path}")
