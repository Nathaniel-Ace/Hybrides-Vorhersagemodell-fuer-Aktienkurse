import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Einstellungen
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025.csv"

# 1) Daten einlesen und Log-Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) GARCH auf gesamten Datensatz einmalig fitten und Volatilität speichern
g      = arch_model(df["Return"] * 10, mean="Zero", vol="GARCH", p=1, q=1,
                    dist="normal", rescale=False)
res_all = g.fit(disp="off")
df["GARCH_vol"] = res_all.conditional_volatility / 10

# 3) Statische Feature-Spalten ermitteln und skalieren
gtd_cols    = [c for c in df.columns if "stock" in c.lower()]
static_cols = ["GARCH_vol", "RSI_14"] + gtd_cols
print("Verwendete statische Features (GTD + RSI + GARCH):", static_cols)

scaler = StandardScaler()
df[static_cols] = scaler.fit_transform(df[static_cols])

# 4) Funktion zum Erzeugen von X, y
def make_xy(subdf):
    X, y = [], []
    rets = subdf["Return"].values
    for i in range(window_size, len(subdf)):
        seq  = list(rets[i-window_size:i])                   # letzte Log-Renditen
        stat = subdf[static_cols].iloc[i].values.tolist()     # GARCH_vol, RSI, GTD
        X.append(seq + stat)
        y.append(rets[i])
    return np.array(X), np.array(y)

# 5) Hyperparameter-Suche für XGBoost
X_full, y_full = make_xy(df)

tscv_search = TimeSeriesSplit(n_splits=n_splits)
param_dist = {
    "n_estimators":     [50, 100, 200],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma":            [0, 0.1, 0.5]
}

# tree_method="hist" für CPU-Training, n_jobs=1 um GPU-Parallelität auszuschließen
xgb = XGBRegressor(
    random_state=42,
    tree_method="hist",
    n_jobs=1
)

search = RandomizedSearchCV(
    xgb,
    param_dist,
    n_iter=20,
    cv=tscv_search,
    scoring="neg_root_mean_squared_error",
    n_jobs=1,            # nur ein Job, um CUDA-Konflikte zu vermeiden
    random_state=42,
    verbose=1,
    error_score='raise'
)
search.fit(X_full, y_full)
best_params = search.best_params_
print(">>> Best XGBoost Params:", best_params)

# 6) Out-of-Sample-Evaluation & Metriken
tscv = TimeSeriesSplit(n_splits=n_splits)
rmse_ret, mae_ret, mape_ret, r2_ret = [], [], [], []
hit_ret, sharpe_ret               = [], []
rmse_prc, mae_prc, mape_prc, r2_prc = [], [], [], []
hit_prc, sharpe_prc               = [], []

fold_results = []  # für Plots aller Folds

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # a) GARCH-Forecast für Test-Set
    fc = res_all.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # b) Skalierung der statischen Features
    train_df[static_cols] = scaler.transform(train_df[static_cols])
    test_df[static_cols]  = scaler.transform(test_df[static_cols])

    # c) Features/Targets
    X_train, y_train = make_xy(train_df)
    X_test,  y_test  = make_xy(test_df)
    print(f"Fold {fold}: X_train={X_train.shape}, X_test={X_test.shape}")

    # d) Model trainieren mit Early Stopping (ebenfalls CPU-hist)
    model = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method="hist",
        n_jobs=1,
        early_stopping_rounds=10,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # e) Log-Return Metriken
    y_pred = model.predict(X_test)
    rm = math.sqrt(mean_squared_error(y_test, y_pred))
    ma = mean_absolute_error(y_test, y_pred)
    mask_r = y_test != 0
    mp = np.mean(np.abs((y_test[mask_r] - y_pred[mask_r]) / y_test[mask_r])) * 100
    r2 = r2_score(y_test, y_pred)
    dir_true = np.sign(np.diff(y_test))
    dir_pred = np.sign(np.diff(y_pred))
    hr = (dir_true == dir_pred).mean() * 100
    rr = np.diff(y_pred) / y_pred[:-1]
    sr = rr.mean() / (rr.std() if rr.std() != 0 else np.nan)

    rmse_ret.append(rm); mae_ret.append(ma)
    mape_ret.append(mp); r2_ret.append(r2)
    hit_ret.append(hr); sharpe_ret.append(sr)

    # f) Preis-Metriken
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iat[i + window_size - 1]
        preds.append(p0 * np.exp(r))
        actuals.append(test_df["Close"].iat[i + window_size])
    preds   = np.array(preds)
    actuals = np.array(actuals)

    rm_p  = math.sqrt(mean_squared_error(actuals, preds))
    ma_p  = mean_absolute_error(actuals, preds)
    mask_p = actuals != 0
    mp_p = np.mean(np.abs((actuals[mask_p] - preds[mask_p]) / actuals[mask_p])) * 100
    r2_p = r2_score(actuals, preds)
    dir_tp = np.sign(np.diff(actuals))
    dir_pp = np.sign(np.diff(preds))
    hr_p = (dir_tp == dir_pp).mean() * 100
    rp = np.diff(preds) / preds[:-1]
    sr_p = rp.mean() / (rp.std() if rp.std() != 0 else np.nan)

    rmse_prc.append(rm_p); mae_prc.append(ma_p)
    mape_prc.append(mp_p); r2_prc.append(r2_p)
    hit_prc.append(hr_p); sharpe_prc.append(sr_p)

    print(f"Fold {fold}:")
    print(f"  Returns → RMSE={rm:.4f}, MAE={ma:.4f}, MAPE={mp:.2f}%, "
          f"R²={r2:.4f}, Hit-Rate={hr:.1f}%, Sharpe={sr:.4f}")
    print(f"  Prices  → RMSE={rm_p:.4f}, MAE={ma_p:.4f}, MAPE={mp_p:.2f}%, "
          f"R²={r2_p:.4f}, Hit-Rate={hr_p:.1f}%, Sharpe={sr_p:.4f}\n")

    # Ergebnisse für spätere Plots speichern
    fold_results.append({
        "idx":    test_df.index[window_size:],
        "y_test": y_test,
        "y_pred": y_pred,
        "actual": actuals,
        "preds":  preds
    })

# 7) Durchschnittsergebnisse ausgeben
print("\n=== Durchschnittliche Metriken: Log-Renditen ===")
print(f"RMSE    = {np.mean(rmse_ret):.4f}")
print(f"MAE     = {np.mean(mae_ret):.4f}")
print(f"MAPE    = {np.mean(mape_ret):.2f}%")
print(f"R²      = {np.mean(r2_ret):.4f}")
print(f"HitRate = {np.mean(hit_ret):.2f}%")
print(f"Sharpe  = {np.nanmean(sharpe_ret):.4f}")

print("\n=== Durchschnittliche Metriken: Preise ===")
print(f"RMSE    = {np.mean(rmse_prc):.4f}")
print(f"MAE     = {np.mean(mae_prc):.4f}")
print(f"MAPE    = {np.mean(mape_prc):.2f}%")
print(f"R²      = {np.mean(r2_prc):.4f}")
print(f"HitRate = {np.mean(hit_prc):.2f}%")
print(f"Sharpe  = {np.nanmean(sharpe_prc):.4f}")

# 8) Plots für alle Folds

# a) Log-Renditen über alle Folds
plt.figure(figsize=(12, 5))
for i, fr in enumerate(fold_results, start=1):
    plt.plot(fr["idx"], fr["y_test"],  label=f"Real Ret Fold {i}",  alpha=0.7)
    plt.plot(fr["idx"], fr["y_pred"], "--",               label=f"Pred Ret Fold {i}", alpha=0.7)
plt.title(f"{ticker} – Log-Renditen über alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log-Rendite")
plt.legend(); plt.tight_layout(); plt.show()

# b) Close-Preise über alle Folds
plt.figure(figsize=(12, 5))
for i, fr in enumerate(fold_results, start=1):
    plt.plot(fr["idx"], fr["actual"], label=f"Real Price Fold {i}", alpha=0.7)
    plt.plot(fr["idx"], fr["preds"],  "--",               label=f"Pred Price Fold {i}", alpha=0.7)
plt.title(f"{ticker} – Close-Preise über alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)")
plt.legend(); plt.tight_layout(); plt.show()
