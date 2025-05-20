import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Einstellungen
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025_with_trends.csv"

# 1) Daten einlesen und Return + GARCH_vol berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# GARCH einmalig auf allen Daten fitten
scaled_all = df["Return"] * 10
garch_all  = arch_model(scaled_all, mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = garch_all.conditional_volatility / 10

# 2) Statistische Features skalieren & GTD-Spalten identifizieren
static_cols = ["GARCH_vol", "RSI_14", "Trend_Average", "Trend_Smoothed"]
print("Verwendete statische Features:", static_cols)

scaler = StandardScaler()
df[static_cols] = scaler.fit_transform(df[static_cols])

# 3) X/y Erzeuger
def make_xy(df):
    X, y = [], []
    for i in range(window_size, len(df)):
        seq  = df["Return"].iloc[i-window_size:i].tolist()
        stat = df[static_cols].iloc[i].tolist()
        X.append(seq + stat)
        y.append(df["Return"].iat[i])
    return np.array(X), np.array(y)

# 4) Hyperparameter‑Grid
param_grid = {
    "units":      [50],
    "dropout":    [0.2],
    "lr":         [1e-3, 5e-4],
    "batch_size": [16]
}

tscv = TimeSeriesSplit(n_splits=n_splits)
best_rmse, best_cfg = np.inf, None

# 5) Grid‑Search über Log‑Return‑RMSE
for units in param_grid["units"]:
    for drop in param_grid["dropout"]:
        for lr in param_grid["lr"]:
            for bs in param_grid["batch_size"]:
                cv_rmses = []
                for tr_idx, te_idx in tscv.split(df):
                    train_df = df.iloc[tr_idx]
                    test_df  = df.iloc[te_idx]
                    X_tr, y_tr = make_xy(train_df)
                    X_ts, y_ts = make_xy(test_df)

                    # Split Train/Val
                    cut = int(len(X_tr) * 0.9)
                    X_train, X_val = X_tr[:cut], X_tr[cut:]
                    y_train, y_val = y_tr[:cut], y_tr[cut:]

                    # reshape
                    X_train = X_train.reshape((-1, X_train.shape[1], 1))
                    X_val   = X_val.reshape((-1, X_val.shape[1], 1))
                    X_test  = X_ts .reshape((-1, X_ts .shape[1], 1))

                    # Modell
                    model = Sequential([
                        Input(shape=(X_train.shape[1],1)),
                        LSTM(units, return_sequences=True),
                        Dropout(drop),
                        LSTM(units),
                        Dropout(drop),
                        Dense(1)
                    ])
                    opt = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(optimizer=opt, loss="mse")
                    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
                    model.fit(X_train, y_train,
                              validation_data=(X_val, y_val),
                              epochs=20, batch_size=bs,
                              callbacks=[es], verbose=0)

                    # Evaluation
                    y_pred = model.predict(X_test).flatten()
                    cv_rmses.append(math.sqrt(mean_squared_error(y_ts, y_pred)))

                avg_rmse = np.mean(cv_rmses)
                if avg_rmse < best_rmse:
                    best_rmse, best_cfg = avg_rmse, {
                        "units":units, "dropout":drop,
                        "lr":lr, "batch_size":bs
                    }

print("\nBest CV‑RMSE (Log‑Renditen):", best_rmse)
print("Best Config:", best_cfg)

# 6) Endgültiges Training & pro-Fold‑RMSEs ausgeben
fold_rmse_ret = []
fold_rmse_prc = []
fold_results  = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), 1):
    train_df = df.iloc[tr_idx]
    test_df  = df.iloc[te_idx]
    X_tr, y_tr = make_xy(train_df)
    X_ts, y_ts = make_xy(test_df)

    # reshape
    X_tr = X_tr.reshape((-1, X_tr.shape[1],1))
    X_ts = X_ts.reshape((-1, X_ts.shape[1],1))

    model = Sequential([
        Input(shape=(X_tr.shape[1],1)),
        LSTM(best_cfg["units"], return_sequences=True),
        Dropout(best_cfg["dropout"]),
        LSTM(best_cfg["units"]),
        Dropout(best_cfg["dropout"]),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(best_cfg["lr"]), loss="mse")
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=20,
              batch_size=best_cfg["batch_size"],
              callbacks=[es], verbose=0)

    # Vorhersage
    y_pred = model.predict(X_ts).flatten()
    rm_ret = math.sqrt(mean_squared_error(y_ts, y_pred))
    fold_rmse_ret.append(rm_ret)

    # Rückrechnung Kurse
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iloc[i+window_size-1]
        preds.append(p0 * np.exp(r))
        actuals.append(test_df["Close"].iloc[i+window_size])
    rm_prc = math.sqrt(mean_squared_error(actuals, preds))
    fold_rmse_prc.append(rm_prc)

    print(f"Fold {fold}: RMSE Log‐Renditen = {rm_ret:.4f}, RMSE Preise = {rm_prc:.4f}")

    fold_results.append({
        "idx":      test_df.index[window_size:],
        "y_test":   y_ts,
        "y_pred":   y_pred,
        "actuals":  actuals,
        "preds":    preds
    })

# Durchschnitt über alle Folds
print(f"\nDurchschn. RMSE Log‑Renditen: {np.mean(fold_rmse_ret):.4f}")
print(f"Durchschn. RMSE Aktienkurse: {np.mean(fold_rmse_prc):.4f}")

# 7) Plots: alle Folds
plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, 1):
    plt.plot(fr["idx"], fr["y_test"],  label=f"Real Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["y_pred"],  "--", label=f"Pred Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Log‑Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log‑Rendite")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, 1):
    plt.plot(fr["idx"], fr["actuals"], label=f"Real Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["preds"],   "--", label=f"Pred Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Aktienkurse alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)")
plt.legend(); plt.tight_layout(); plt.show()
