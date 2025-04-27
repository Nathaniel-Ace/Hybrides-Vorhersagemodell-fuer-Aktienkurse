import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 1) Daten einlesen & Renditen berechnen
ticker = "NVDA"
df = pd.read_csv(f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
                 parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Cross-Validation Setup
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# 3) Listen für Metriken (Returns)
rmse_ret, mae_ret, mape_ret, r2_ret = [], [], [], []
hit_ret, sharpe_ret = [], []

# 4) Listen für Metriken (Prices)
rmse_pr, mae_pr, mape_pr, r2_pr = [], [], [], []
hit_pr, sharpe_pr = [], []

# Platzhalter für letzten Fold
last = {}

window_size = 10
plot_fold = n_splits
fold = 1

def create_features_and_target(df, window_size=10):
    X, y = [], []
    returns = df["Return"].values
    vol = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        X.append(list(returns[i-window_size:i]) + [vol[i]])
        y.append(returns[i])
    return np.array(X), np.array(y)

for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # GARCH fitten
    scaled = train_df["Return"] * 10
    gm = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = gm.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Features / Targets
    X_tr, y_tr = create_features_and_target(train_df, window_size)
    X_te, y_te = create_features_and_target(test_df, window_size)

    # reshape für LSTM [Samples, TimeSteps, Features]
    X_tr_l = X_tr.reshape(-1, window_size+1, 1)
    X_te_l = X_te.reshape(-1, window_size+1, 1)

    # LSTM definieren & trainieren
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

    model = Sequential([
        Input(shape=(window_size+1, 1)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(1)
    ])
    model.compile("adam", "mse")
    model.fit(X_tr_l, y_tr, epochs=20, batch_size=16, verbose=0)

    # Vorhersage Returns
    y_pred = model.predict(X_te_l).flatten()

    # Metriken auf Returns
    rm = math.sqrt(mean_squared_error(y_te, y_pred))
    ma = mean_absolute_error(y_te, y_pred)
    # MAPE: Nullwerte ausklammern
    denom_ret = np.where(y_te == 0, np.nan, y_te)
    mp = np.nanmean(np.abs((y_te - y_pred) / denom_ret)) * 100
    r2 = r2_score(y_te, y_pred)
    dir_true = np.sign(np.diff(y_te))
    dir_pred = np.sign(np.diff(y_pred))
    hr = (dir_true == dir_pred).mean() * 100
    ret_ret = np.diff(y_pred) / y_pred[:-1]
    sr = ret_ret.mean() / (ret_ret.std() if ret_ret.std() != 0 else np.nan)

    rmse_ret.append(rm);
    mae_ret.append(ma)
    mape_ret.append(mp);
    r2_ret.append(r2)
    hit_ret.append(hr);
    sharpe_ret.append(sr)

    # Rückrechnung in Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        prev = test_df["Close"].iloc[i + window_size - 1]
        p = prev * np.exp(r)
        preds.append(p)
        actuals.append(test_df["Close"].iloc[i + window_size])

    preds = np.array(preds)
    actuals = np.array(actuals)
    # Metriken auf Prices
    rm_p = math.sqrt(mean_squared_error(actuals, preds))
    ma_p = mean_absolute_error(actuals, preds)
    denom_pr = np.where(actuals == 0, np.nan, actuals)
    mp_p = np.nanmean(np.abs((actuals - preds) / denom_pr)) * 100
    r2_p = r2_score(actuals, preds)
    dir_t = np.sign(np.diff(actuals))
    dir_p = np.sign(np.diff(preds))
    hr_p = (dir_t == dir_p).mean() * 100
    ret_pr = np.diff(preds) / preds[:-1]
    sr_p = ret_pr.mean() / (ret_pr.std() if ret_pr.std() != 0 else np.nan)

    rmse_pr.append(rm_p);
    mae_pr.append(ma_p)
    mape_pr.append(mp_p);
    r2_pr.append(r2_p)
    hit_pr.append(hr_p);
    sharpe_pr.append(sr_p)

    # letzten Fold merken
    if fold == plot_fold:
        last = {
            "y_te": y_te, "y_pred": y_pred,
            "actuals": actuals, "preds": preds
        }
    fold += 1

# Ausgabe Durchschnittswerte
print(f"=== Returns (avg over {n_splits} folds) ===")
print(f"RMSE  {np.mean(rmse_ret):.4f}, MAE  {np.mean(mae_ret):.4f}, MAPE  {np.mean(mape_ret):.2f}%, "
      f"R²  {np.mean(r2_ret):.4f}, HitRate  {np.mean(hit_ret):.1f}%, Sharpe  {np.mean(sharpe_ret):.4f}")
print(f"=== Prices (avg over {n_splits} folds) ===")
print(f"RMSE  {np.mean(rmse_pr):.4f}, MAE  {np.mean(mae_pr):.4f}, MAPE  {np.mean(mape_pr):.2f}%, "
      f"R²  {np.mean(r2_pr):.4f}, HitRate  {np.mean(hit_pr):.1f}%, Sharpe  {np.mean(sharpe_pr):.4f}")

# ————— Plots für letzten Fold —————
plt.figure(figsize=(12,5))
plt.plot(last["y_te"], label="Actual Returns")
plt.plot(last["y_pred"], label="Predicted Returns", alpha=0.7)
plt.title(f"{ticker} – Log-Returns (Fold {n_splits})")
plt.xlabel("Sample"); plt.ylabel("Log-Return"); plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(last["actuals"], label="Actual Close")
plt.plot(last["preds"],   label="Predicted Close", alpha=0.7)
plt.title(f"{ticker} – Close Price (Fold {n_splits})")
plt.xlabel("Sample"); plt.ylabel("Price"); plt.legend()
plt.show()
