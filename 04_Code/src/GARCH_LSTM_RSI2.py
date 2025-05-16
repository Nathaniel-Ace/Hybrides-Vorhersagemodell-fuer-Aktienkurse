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
ticker      = "MSFT"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Return + GARCH_vol berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# GARCH einmalig auf allen Daten fitten
scaled_all = df["Return"] * 10
garch_all  = arch_model(scaled_all, mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = garch_all.conditional_volatility / 10

# 2) Statistische Features skalieren (RSI + GARCH_vol)
static_cols = ["GARCH_vol", "RSI_14"]
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

# 4) Hyperparameter‑Grid und CV-Setup
param_grid = {"units":[50], "dropout":[0.2], "lr":[1e-3, 5e-4], "batch_size":[16]}
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
                    cut = int(len(X_tr)*0.9)
                    X_train, X_val = X_tr[:cut], X_tr[cut:]
                    y_train, y_val = y_tr[:cut], y_tr[cut:]
                    X_train = X_train.reshape((-1, X_train.shape[1],1))
                    X_val   = X_val.reshape((-1, X_val.shape[1],1))
                    X_test  = X_ts .reshape((-1, X_ts .shape[1],1))
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
                    model.fit(X_train, y_train, validation_data=(X_val,y_val),
                              epochs=20, batch_size=bs, callbacks=[es], verbose=0)
                    y_pred = model.predict(X_test).flatten()
                    cv_rmses.append(math.sqrt(mean_squared_error(y_ts, y_pred)))
                avg_rmse = np.mean(cv_rmses)
                if avg_rmse < best_rmse:
                    best_rmse, best_cfg = avg_rmse, {"units":units,"dropout":drop,"lr":lr,"batch_size":bs}
print(f"\nBest CV‑RMSE (Log‑Renditen): {best_rmse:.4f}")
print("Best Config:", best_cfg)

# 5) Finales Modelltraining auf dem gesamten Datensatz
X_all, y_all = make_xy(df)
X_all = X_all.reshape((-1, X_all.shape[1], 1))

model = Sequential([
    Input(shape=(X_all.shape[1], 1)),
    LSTM(best_cfg["units"], return_sequences=True),
    Dropout(best_cfg["dropout"]),
    LSTM(best_cfg["units"]),
    Dropout(best_cfg["dropout"]),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(best_cfg["lr"]), loss="mse")
model.fit(X_all, y_all, epochs=20, batch_size=best_cfg["batch_size"], verbose=1)

# Modellperformance auf Trainingdaten berechnen
y_pred_all = model.predict(X_all).flatten()

# RMSE auf Renditeebene (Train)
rmse_ret_all = math.sqrt(mean_squared_error(y_all, y_pred_all))
print(f"\n📈 RMSE (Renditeebene) auf Trainingsdaten: {rmse_ret_all:.4f}")

# RMSE auf Preisebene rekonstruieren
# Schritt 1: Startpreis
start_prices = df["Close"].iloc[window_size - 1 : -1].values
true_prices = df["Close"].iloc[window_size:].values
pred_prices = start_prices * np.exp(y_pred_all)

# Schritt 2: RMSE auf Preisbasis
rmse_prc_all = math.sqrt(mean_squared_error(true_prices, pred_prices))
print(f"💰 RMSE (Preisebene) auf Trainingsdaten:  {rmse_prc_all:.4f}")

# Modell speichern
model.save(f"../../05_Modelle/garch_lstm_{ticker.lower()}_final_model.keras")
print("✅ Finales Modell gespeichert")


# 6) Endgültiges Training & Metriken pro Fold ausgeben
metrics = {"rmse_ret":[],"mae_ret":[],"mape_ret":[],"r2_ret":[],"hit_ret":[],"sharpe_ret":[],
           "rmse_prc":[],"mae_prc":[],"mape_prc":[],"r2_prc":[],"hit_prc":[],"sharpe_prc":[]}
fold_results = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), 1):
    train_df = df.iloc[tr_idx]
    test_df  = df.iloc[te_idx]
    X_tr, y_tr = make_xy(train_df)
    X_ts, y_ts = make_xy(test_df)
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
    model.fit(X_tr, y_tr, epochs=20, batch_size=best_cfg["batch_size"], callbacks=[es], verbose=0)
    y_pred = model.predict(X_ts).flatten()

    # Returns-Metriken
    rm_ret = math.sqrt(mean_squared_error(y_ts, y_pred))
    mae_ret = mean_absolute_error(y_ts, y_pred)
    denom_r = np.where(y_ts==0, np.nan, y_ts)
    mape_ret = np.nanmean(np.abs((y_ts - y_pred)/denom_r))*100
    r2_ret = r2_score(y_ts, y_pred)
    dir_true = np.sign(np.diff(y_ts))
    dir_pred = np.sign(np.diff(y_pred))
    hit_ret = (dir_true==dir_pred).mean()*100
    ret_ret = np.diff(y_pred)/y_pred[:-1]
    sharpe_ret = ret_ret.mean()/(ret_ret.std() if ret_ret.std()!=0 else np.nan)
    metrics["rmse_ret"].append(rm_ret)
    metrics["mae_ret"].append(mae_ret)
    metrics["mape_ret"].append(mape_ret)
    metrics["r2_ret"].append(r2_ret)
    metrics["hit_ret"].append(hit_ret)
    metrics["sharpe_ret"].append(sharpe_ret)

    # Preis-Metriken
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iloc[i+window_size-1]
        p = p0 * np.exp(r)
        preds.append(p)
        actuals.append(test_df["Close"].iloc[i+window_size])
    preds = np.array(preds); actuals = np.array(actuals)
    rm_prc = math.sqrt(mean_squared_error(actuals, preds))
    mae_prc = mean_absolute_error(actuals, preds)
    denom_p = np.where(actuals==0, np.nan, actuals)
    mape_prc = np.nanmean(np.abs((actuals - preds)/denom_p))*100
    r2_prc = r2_score(actuals, preds)
    dir_t = np.sign(np.diff(actuals))
    dir_p = np.sign(np.diff(preds))
    hit_prc = (dir_t==dir_p).mean()*100
    ret_pr = np.diff(preds)/preds[:-1]
    sharpe_prc = ret_pr.mean()/(ret_pr.std() if ret_pr.std()!=0 else np.nan)
    metrics["rmse_prc"].append(rm_prc)
    metrics["mae_prc"].append(mae_prc)
    metrics["mape_prc"].append(mape_prc)
    metrics["r2_prc"].append(r2_prc)
    metrics["hit_prc"].append(hit_prc)
    metrics["sharpe_prc"].append(sharpe_prc)
    print(f"Fold {fold}:\n"
          f"  Returns → RMSE={rm_ret:.4f}, MAE={mae_ret:.4f}, MAPE={mape_ret:.4f}%, R2={r2_ret:.4f}, Hit-Rate={hit_ret:.4f}%, Sharpe={sharpe_ret:.4f}\n"
          f"  Prices  → RMSE={rm_prc:.4f}, MAE={mae_prc:.4f}, MAPE={mape_prc:.4f}%, R2={r2_prc:.4f}, Hit-Rate={hit_prc:.4f}%, Sharpe={sharpe_prc:.4f}")
    fold_results.append({"idx":test_df.index[window_size:], "y_test":y_ts, "y_pred":y_pred,
                         "actuals":actuals, "preds":preds})

# 7) Durchschnittliche Metriken über alle Folds
# berechne Mittelwerte
avg_rmse_ret   = np.nanmean(metrics["rmse_ret"])
avg_mae_ret    = np.nanmean(metrics["mae_ret"])
avg_mape_ret   = np.nanmean(metrics["mape_ret"])
avg_r2_ret     = np.nanmean(metrics["r2_ret"])
avg_hit_ret    = np.nanmean(metrics["hit_ret"])
avg_sharpe_ret = np.nanmean(metrics["sharpe_ret"])

avg_rmse_prc   = np.nanmean(metrics["rmse_prc"])
avg_mae_prc    = np.nanmean(metrics["mae_prc"])
avg_mape_prc   = np.nanmean(metrics["mape_prc"])
avg_r2_prc     = np.nanmean(metrics["r2_prc"])
avg_hit_prc    = np.nanmean(metrics["hit_prc"])
avg_sharpe_prc = np.nanmean(metrics["sharpe_prc"])

# Ausgabe
print("\n=== Durchschnittliche Metriken: Log-Renditen ===")
print(f"RMSE      = {avg_rmse_ret:.4f}")
print(f"MAE       = {avg_mae_ret:.4f}")
print(f"MAPE      = {avg_mape_ret:,.4f}%")
print(f"R²        = {avg_r2_ret:.4f}")
print(f"HitRate   = {avg_hit_ret:.4f}%")
print(f"Sharpe    = {avg_sharpe_ret:.4f}")

print("\n=== Durchschnittliche Metriken: Preise ===")
print(f"RMSE      = {avg_rmse_prc:.4f}")
print(f"MAE       = {avg_mae_prc:.4f}")
print(f"MAPE      = {avg_mape_prc:.4f}%")
print(f"R²        = {avg_r2_prc:.4f}")
print(f"HitRate   = {avg_hit_prc:.4f}%")
print(f"Sharpe    = {avg_sharpe_prc:.4f}")

# 8) Plots: Log-Renditen & Preise über alle Folds
plt.figure(figsize=(12,5))
for i,fr in enumerate(fold_results,1):
    plt.plot(fr["idx"], fr["y_test"], label=f"Real Ret Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["y_pred"], "--", label=f"Pred Ret Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Log‑Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log‑Rendite"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,5))
for i,fr in enumerate(fold_results,1):
    plt.plot(fr["idx"], fr["actuals"], label=f"Real Prc Fold {i}", alpha=0.8)
    plt.plot(fr["idx"], fr["preds"],   "--", label=f"Pred Prc Fold {i}", alpha=0.8)
plt.title(f"{ticker} – Aktienkurse alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.tight_layout(); plt.show()
