import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Einstellungen
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025.csv"

# 1) Daten einlesen und Log-Rendite berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) GARCH einmalig auf gesamten Datensatz fitten und volatilität speichern
scaled_all = df["Return"] * 10
garch_all  = arch_model(
    scaled_all, mean="Zero", vol="GARCH", p=1, q=1,
    dist="normal", rescale=False
).fit(disp="off")
df["GARCH_vol"] = garch_all.conditional_volatility / 10

# 3) Statistische Features skalieren & GTD-Spalten identifizieren
static_cols = ["GARCH_vol", "RSI_14"] + [col for col in df.columns if "stock" in col.lower()]
print("Verwendete statische Features (GTD + RSI + GARCH):", static_cols)

scaler = StandardScaler()
df[static_cols] = scaler.fit_transform(df[static_cols])

# 4) X/y-Erzeuger
def make_xy(data: pd.DataFrame):
    X, y = [], []
    for i in range(window_size, len(data)):
        seq  = data["Return"].iloc[i-window_size:i].tolist()
        stat = data[static_cols].iloc[i].tolist()
        X.append(seq + stat)
        y.append(data["Return"].iat[i])
    return np.array(X), np.array(y)

# 5) Hyperparameter-Grid und CV-Setup
param_grid = {
    "units":      [50],
    "dropout":    [0.2],
    "lr":         [1e-3, 5e-4],
    "batch_size": [16]
}
tscv = TimeSeriesSplit(n_splits=n_splits)
best_rmse, best_cfg = np.inf, None

# 6) Grid-Search über Log-Return-RMSE
for units in param_grid["units"]:
    for drop in param_grid["dropout"]:
        for lr in param_grid["lr"]:
            for bs in param_grid["batch_size"]:
                cv_rmses = []
                for tr_idx, te_idx in tscv.split(df):
                    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
                    X_tr, y_tr = make_xy(tr)
                    X_te, y_te = make_xy(te)
                    # Train/Val-Split
                    cut = int(len(X_tr)*0.9)
                    X_train, X_val = X_tr[:cut], X_tr[cut:]
                    y_train, y_val = y_tr[:cut], y_tr[cut:]
                    # Reshape
                    X_train = X_train.reshape(-1, X_train.shape[1], 1)
                    X_val   = X_val.reshape(-1, X_val.shape[1], 1)
                    X_test  = X_te .reshape(-1, X_te .shape[1], 1)
                    # Modell
                    model = Sequential([
                        Input(shape=(X_train.shape[1],1)),
                        LSTM(units, return_sequences=True),
                        Dropout(drop),
                        LSTM(units),
                        Dropout(drop),
                        Dense(1)
                    ])
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss="mse"
                    )
                    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val,y_val),
                        epochs=20, batch_size=bs,
                        callbacks=[es], verbose=0
                    )
                    y_pred = model.predict(X_test).flatten()
                    cv_rmses.append(math.sqrt(mean_squared_error(y_te, y_pred)))
                avg = np.mean(cv_rmses)
                if avg < best_rmse:
                    best_rmse, best_cfg = avg, {"units":units,"dropout":drop,"lr":lr,"batch_size":bs}

print(f"\nBest CV-RMSE (Log-Renditen): {best_rmse:.4f}")
print("Best Config:", best_cfg)

# 7) Endgültiges Training & Metriken pro Fold
metrics = {
    "rmse_ret":[], "mae_ret":[], "mape_ret":[], "r2_ret":[], "hit_ret":[], "sharpe_ret":[],
    "rmse_prc":[], "mae_prc":[], "mape_prc":[], "r2_prc":[], "hit_prc":[], "sharpe_prc":[]
}
fold_results = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
    tr = df.iloc[tr_idx]
    te = df.iloc[te_idx].copy()

    # GARCH-Forecast für Test-Set
    fc = garch_all.forecast(
        start=tr.index[-1],
        horizon=len(te),
        reindex=False
    )
    te["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10
    # Skaliere die statischen Features in te
    te[static_cols] = scaler.transform(te[static_cols])

    # Features / Targets
    X_tr, y_tr = make_xy(tr)
    X_te, y_te = make_xy(te)

    # Reshape
    X_train = X_tr.reshape(-1, X_tr.shape[1], 1)
    X_test  = X_te.reshape(-1, X_te.shape[1], 1)

    # Modell trainieren
    model = Sequential([
        Input(shape=(X_train.shape[1],1)),
        LSTM(best_cfg["units"], return_sequences=True),
        Dropout(best_cfg["dropout"]),
        LSTM(best_cfg["units"]),
        Dropout(best_cfg["dropout"]),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(best_cfg["lr"]),
        loss="mse"
    )
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_tr,
        epochs=20, batch_size=best_cfg["batch_size"],
        callbacks=[es], verbose=0
    )

    # Vorhersage
    y_pred = model.predict(X_test).flatten()

    # — Returns-Metriken
    rm_ret = math.sqrt(mean_squared_error(y_te, y_pred))
    mae_ret = mean_absolute_error(y_te, y_pred)
    mask_r = y_te != 0
    mape_ret = np.mean(np.abs((y_te[mask_r] - y_pred[mask_r]) / y_te[mask_r])) * 100
    r2_ret = r2_score(y_te, y_pred)
    dir_true = np.sign(np.diff(y_te))
    dir_pred = np.sign(np.diff(y_pred))
    hit_ret = (dir_true == dir_pred).mean() * 100
    ret_ret = np.diff(y_pred) / y_pred[:-1]
    sharpe_ret = ret_ret.mean() / (ret_ret.std() if ret_ret.std() != 0 else np.nan)

    # — Price-Metriken
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = te["Close"].iat[i + window_size - 1]
        preds.append(p0 * np.exp(r))
        actuals.append(te["Close"].iat[i + window_size])
    preds = np.array(preds)
    actuals = np.array(actuals)

    rm_prc = math.sqrt(mean_squared_error(actuals, preds))
    mae_prc = mean_absolute_error(actuals, preds)
    mask_p = actuals != 0
    mape_prc = np.mean(np.abs((actuals[mask_p] - preds[mask_p]) / actuals[mask_p])) * 100
    r2_prc = r2_score(actuals, preds)
    dir_t = np.sign(np.diff(actuals))
    dir_p = np.sign(np.diff(preds))
    hit_prc = (dir_t == dir_p).mean() * 100
    ret_pr = np.diff(preds) / preds[:-1]
    sharpe_prc = ret_pr.mean() / (ret_pr.std() if ret_pr.std() != 0 else np.nan)

    # Speichern
    for k, v in [
        ("rmse_ret", rm_ret), ("mae_ret", mae_ret), ("mape_ret", mape_ret),
        ("r2_ret", r2_ret), ("hit_ret", hit_ret), ("sharpe_ret", sharpe_ret),
        ("rmse_prc", rm_prc), ("mae_prc", mae_prc), ("mape_prc", mape_prc),
        ("r2_prc", r2_prc), ("hit_prc", hit_prc), ("sharpe_prc", sharpe_prc),
    ]:
        metrics[k].append(v)

    print(f"Fold {fold}:")
    print(f"  Returns → RMSE={rm_ret:.4f}, MAE={mae_ret:.4f}, MAPE={mape_ret:.2f}%, "
          f"R²={r2_ret:.4f}, Hit-Rate={hit_ret:.1f}%, Sharpe={sharpe_ret:.4f}")
    print(f"  Prices  → RMSE={rm_prc:.4f}, MAE={mae_prc:.4f}, MAPE={mape_prc:.2f}%, "
          f"R²={r2_prc:.4f}, Hit-Rate={hit_prc:.1f}%, Sharpe={sharpe_prc:.4f}\n")

    fold_results.append({
        "idx":      te.index[window_size:],
        "y_test":   y_te,
        "y_pred":   y_pred,
        "actuals":  actuals,
        "preds":    preds
    })

# 8) Durchschnittliche Metriken
print("\n=== Durchschnittliche Metriken: Log-Renditen ===")
print(f"RMSE    = {np.nanmean(metrics['rmse_ret']):.4f}")
print(f"MAE     = {np.nanmean(metrics['mae_ret']):.4f}")
print(f"MAPE    = {np.nanmean(metrics['mape_ret']):.2f}%")
print(f"R²      = {np.nanmean(metrics['r2_ret']):.4f}")
print(f"HitRate = {np.nanmean(metrics['hit_ret']):.2f}%")
print(f"Sharpe  = {np.nanmean(metrics['sharpe_ret']):.4f}")

print("\n=== Durchschnittliche Metriken: Preise ===")
print(f"RMSE    = {np.nanmean(metrics['rmse_prc']):.4f}")
print(f"MAE     = {np.nanmean(metrics['mae_prc']):.4f}")
print(f"MAPE    = {np.nanmean(metrics['mape_prc']):.2f}%")
print(f"R²      = {np.nanmean(metrics['r2_prc']):.4f}")
print(f"HitRate = {np.nanmean(metrics['hit_prc']):.2f}%")
print(f"Sharpe  = {np.nanmean(metrics['sharpe_prc']):.4f}")

# 9) Plots: alle Folds
plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, 1):
    plt.plot(fr["idx"], fr["y_test"],  label=f"Real Ret Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["y_pred"], "--", label=f"Pred Ret Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Log-Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, 1):
    plt.plot(fr["idx"], fr["actuals"], label=f"Real Prc Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["preds"],   "--", label=f"Pred Prc Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Aktienkurse alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.tight_layout(); plt.show()
