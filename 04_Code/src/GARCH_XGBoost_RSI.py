import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Einstellungen
ticker      = "NVDA"
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

# 3) Features/Target erzeugen
def make_xy(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    rsis = df["RSI_14"].values
    for i in range(window_size, len(df)):
        seq   = list(rets[i-window_size:i])
        feats = seq + [vols[i], rsis[i]]
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

X_full, y_full = make_xy(df, window_size)

# 4) Hyperparameter-Suche mit RandomizedSearchCV
param_dist = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "gamma":           [0, 0.1, 0.5]
}
xgb = XGBRegressor(random_state=42, tree_method="hist")
tscv_search = TimeSeriesSplit(n_splits=n_splits)
search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=20,
    cv=tscv_search,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_full, y_full)
best_params = search.best_params_
print(">>> Best XGBoost Params:", best_params)

# 5) Out-of-Sample Evaluation und Metriken
tscv = TimeSeriesSplit(n_splits=n_splits)
rmse_ret, mae_ret, mape_ret, r2_ret = [], [], [], []
hit_ret, sharpe_ret                   = [], []
rmse_prc, mae_prc, mape_prc, r2_prc   = [], [], [], []
hit_prc, sharpe_prc                   = [], []

last_fold = None

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # GARCH-Forecast für Test-Set
    fc = res_all.forecast(
        start=train_df.index[-1],
        horizon=len(test_df),
        reindex=False
    )
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Features & Targets
    X_train, y_train = make_xy(train_df, window_size)
    X_test,  y_test  = make_xy(test_df,  window_size)

    # Train finaler XGB mit besten Parametern
    model = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train, verbose=False)

    # Log-Return Metriken
    y_pred = model.predict(X_test)
    rm = math.sqrt(mean_squared_error(y_test, y_pred))
    ma = mean_absolute_error(y_test, y_pred)
    mask = (y_test != 0)
    mp = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    r2 = r2_score(y_test, y_pred)
    dir_true = np.sign(np.diff(y_test))
    dir_pred = np.sign(np.diff(y_pred))
    hr = (dir_true == dir_pred).mean() * 100
    rr = np.diff(y_pred) / y_pred[:-1]
    sr = rr.mean() / (rr.std() if rr.std()!=0 else np.nan)

    rmse_ret.append(rm); mae_ret.append(ma)
    mape_ret.append(mp); r2_ret.append(r2)
    hit_ret.append(hr); sharpe_ret.append(sr)

    # Price Metriken
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iat[i + window_size - 1]
        preds.append(p0 * np.exp(r))
        actuals.append(test_df["Close"].iat[i + window_size])
    preds = np.array(preds); actuals = np.array(actuals)

    rm_p = math.sqrt(mean_squared_error(actuals, preds))
    ma_p = mean_absolute_error(actuals, preds)
    mp_p = np.mean(np.abs((actuals - preds) /
                          np.where(actuals==0, np.nan, actuals))) * 100
    r2_p = r2_score(actuals, preds)
    dir_tp = np.sign(np.diff(actuals))
    dir_pp = np.sign(np.diff(preds))
    hr_p = (dir_tp == dir_pp).mean() * 100
    rp = np.diff(preds) / preds[:-1]
    sr_p = rp.mean() / (rp.std() if rp.std()!=0 else np.nan)

    rmse_prc.append(rm_p); mae_prc.append(ma_p)
    mape_prc.append(mp_p); r2_prc.append(r2_p)
    hit_prc.append(hr_p); sharpe_prc.append(sr_p)

    print(f"Fold {fold}:")
    print(f"  Returns → RMSE={rm:.4f}, MAE={ma:.4f}, MAPE={mp:.2f}%, "
          f"R²={r2:.4f}, Hit-Rate={hr:.1f}%, Sharpe={sr:.4f}")
    print(f"  Prices  → RMSE={rm_p:.4f}, MAE={ma_p:.4f}, MAPE={mp_p:.2f}%, "
          f"R²={r2_p:.4f}, Hit-Rate={hr_p:.1f}%, Sharpe={sr_p:.4f}\n")

    if fold == n_splits:
        last_fold = {
            "idx":   test_df.index[window_size:],
            "y_t":   y_test,   "y_p":   y_pred,
            "act":   actuals,  "preds": preds
        }

# 6) Durchschnittliche Metriken zusammenfassen
print("\n=== Durchschnittliche Metriken: Log-Renditen ===")
print(f"RMSE      = {np.mean(rmse_ret):.4f}")
print(f"MAE       = {np.mean(mae_ret):.4f}")
print(f"MAPE      = {np.mean(mape_ret):.2f}%")
print(f"R²        = {np.mean(r2_ret):.4f}")
print(f"HitRate   = {np.mean(hit_ret):.2f}%")
print(f"Sharpe    = {np.nanmean(sharpe_ret):.4f}")

print("\n=== Durchschnittliche Metriken: Preise ===")
print(f"RMSE      = {np.mean(rmse_prc):.4f}")
print(f"MAE       = {np.mean(mae_prc):.4f}")
print(f"MAPE      = {np.mean(mape_prc):.2f}%")
print(f"R²        = {np.mean(r2_prc):.4f}")
print(f"HitRate   = {np.mean(hit_prc):.2f}%")
print(f"Sharpe    = {np.nanmean(sharpe_prc):.4f}")

# 7) Plots für den letzten Fold
if last_fold:
    idx = last_fold["idx"]
    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_t"],  label="Real Log-Renditen")
    plt.plot(idx, last_fold["y_p"],  "--", label="Pred Log-Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log-Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend(); plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["act"],   label="Real Kurs")
    plt.plot(idx, last_fold["preds"], "--", label="Pred Kurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.show()
