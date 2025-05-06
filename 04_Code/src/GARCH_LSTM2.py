import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

# 1) Daten einlesen & Renditen berechnen
ticker = "GOOG"
df = pd.read_csv(
    f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
    parse_dates=["Date"], index_col="Date"
).sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) CV-Setup
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
window_size = 10

# 3) Speicher für Metriken
rmse_ret, mae_ret, mape_ret, r2_ret = [], [], [], []
hit_ret, sharpe_ret = [], []
rmse_pr, mae_pr, mape_pr, r2_pr = [], [], [], []
hit_pr, sharpe_pr = [], []

# 4) Speicher für Plots
all_ret_true, all_ret_pred, dates_ret = [], [], []
all_pr_true, all_pr_pred, dates_pr = [], [], []

# Helper zum Feature-Engineering
def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vol  = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        X.append(np.concatenate([rets[i-window_size:i], [vol[i]]]))
        y.append(rets[i])
    return np.array(X), np.array(y)

# 5) Schleife über die Folds
fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # a) GARCH auf den Trainings-Returns fitten
    scaled = train_df["Return"] * 10
    gm = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = gm.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # b) GARCH-Forecast für den Testabschnitt
    horizon = len(test_df)
    fc = res.forecast(start=train_df.index[-1], horizon=horizon, reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # c) Features und Target
    X_tr, y_tr = create_features_and_target(train_df, window_size)
    X_te, y_te = create_features_and_target(test_df,  window_size)

    # d) LSTM definieren & trainieren
    X_tr_l = X_tr.reshape(-1, window_size+1, 1)
    X_te_l = X_te.reshape(-1, window_size+1, 1)
    model = Sequential([
        Input(shape=(window_size+1, 1)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50),                     Dropout(0.2),
        Dense(1)
    ])
    model.compile("adam", "mse")
    model.fit(X_tr_l, y_tr, epochs=20, batch_size=16, verbose=0)

    # e) Vorhersage der Log-Returns
    y_pred = model.predict(X_te_l).flatten()

    # Datums-Vektor (für beide Plots identisch)
    dates = test_df.index[window_size:]
    dates_ret.append(dates)
    all_ret_true.append(y_te)
    all_ret_pred.append(y_pred)

    # f) Metriken auf Returns
    rm = math.sqrt(mean_squared_error(y_te, y_pred))
    ma = mean_absolute_error(y_te, y_pred)
    denom_r = np.where(y_te == 0, np.nan, y_te)
    mp = np.nanmean(np.abs((y_te - y_pred) / denom_r)) * 100
    r2v = r2_score(y_te, y_pred)
    dir_t = np.sign(np.diff(y_te))
    dir_p = np.sign(np.diff(y_pred))
    hr  = (dir_t == dir_p).mean() * 100
    rets_pred = np.diff(y_pred) / y_pred[:-1]
    sr  = rets_pred.mean() / (rets_pred.std() if rets_pred.std() != 0 else np.nan)

    rmse_ret.append(rm); mae_ret.append(ma); mape_ret.append(mp)
    r2_ret.append(r2v); hit_ret.append(hr); sharpe_ret.append(sr)

    # g) Rückrechnung in Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        prev_price = test_df["Close"].iloc[i + window_size - 1]
        preds.append(prev_price * np.exp(r))
        actuals.append(test_df["Close"].iloc[i + window_size])
    preds   = np.array(preds)
    actuals = np.array(actuals)

    dates_pr.append(dates)
    all_pr_true.append(actuals)
    all_pr_pred.append(preds)

    # h) Metriken auf Preise
    rm_p = math.sqrt(mean_squared_error(actuals, preds))
    ma_p = mean_absolute_error(actuals, preds)
    denom_p = np.where(actuals == 0, np.nan, actuals)
    mp_p = np.nanmean(np.abs((actuals - preds) / denom_p)) * 100
    r2p  = r2_score(actuals, preds)
    dt   = np.sign(np.diff(actuals))
    dp   = np.sign(np.diff(preds))
    hr_p = (dt == dp).mean() * 100
    rets_p = np.diff(preds) / preds[:-1]
    sr_p    = rets_p.mean() / (rets_p.std() if rets_p.std() != 0 else np.nan)

    rmse_pr.append(rm_p); mae_pr.append(ma_p); mape_pr.append(mp_p)
    r2_pr.append(r2p);  hit_pr.append(hr_p);  sharpe_pr.append(sr_p)

    print(f"Fold {fold} fertig.")
    fold += 1

# 6) Durchschnittswerte ausgeben
print(f"=== Log-Returns (Ø über {n_splits} Folds) ===")
print(f"RMSE: {np.mean(rmse_ret):.4f}, MAE: {np.mean(mae_ret):.4f}, "
      f"MAPE: {np.mean(mape_ret):.4f}%, R²: {np.mean(r2_ret):.4f}, "
      f"Hit: {np.mean(hit_ret):.4f}%, Sharpe: {np.nanmean(sharpe_ret):.4f}")
print(f"=== Close-Preise (Ø über {n_splits} Folds) ===")
print(f"RMSE: {np.mean(rmse_pr):.4f}, MAE: {np.mean(mae_pr):.4f}, "
      f"MAPE: {np.mean(mape_pr):.4f}%, R²: {np.mean(r2_pr):.4f}, "
      f"Hit: {np.mean(hit_pr):.4f}%, Sharpe: {np.nanmean(sharpe_pr):.4f}")

# 7) Log-Returns aller 3 Folds
plt.figure(figsize=(12,6))
for i in range(n_splits):
    plt.plot(dates_ret[i], all_ret_true[i],  label=f"Real Ret Fold {i+1}")
    plt.plot(dates_ret[i], all_ret_pred[i], linestyle="--", label=f"Pred Ret Fold {i+1}")
plt.title(f"{ticker} – Log-Renditen alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Log-Return")
plt.legend()
plt.tight_layout()
plt.show()

# 8) Close-Preise aller 3 Folds
plt.figure(figsize=(12,6))
for i in range(n_splits):
    plt.plot(dates_pr[i], all_pr_true[i],  label=f"Real Prc Fold {i+1}")
    plt.plot(dates_pr[i], all_pr_pred[i], linestyle="--", label=f"Pred Prc Fold {i+1}")
plt.title(f"{ticker} – Close-Preise alle {n_splits} Folds")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)")
plt.legend()
plt.tight_layout()
plt.show()
