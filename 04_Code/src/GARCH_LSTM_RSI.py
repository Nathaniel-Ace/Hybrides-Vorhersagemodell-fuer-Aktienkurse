import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

# Einstellungen
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits)
rmse_returns = []
rmse_prices  = []

# 3) Feature-Creator: Sequenz aus vergangenen Renditen + aktueller RSI
def create_features_and_target(df, window_size=10):
    X, y = [], []
    for i in range(window_size, len(df)):
        # a) Rolling-Window der letzten `window_size` Log-Renditen
        ret_seq = df["Return"].iloc[i-window_size:i].tolist()
        # b) Nur RSI_14 als statisches Feature zum Zeitpunkt i
        rsi = df["RSI_14"].iat[i]
        # c) zusammenfügen und Ziel definieren
        features = ret_seq + [rsi]
        X.append(features)
        y.append(df["Return"].iat[i])
    return np.array(X), np.array(y)

# 4) Cross-Validation
last_fold = None
fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 5) Features und Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    print(f"Fold {fold}: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")

    # 6) Reshape für LSTM: jeder Zeitschritt hat genau 1 Feature (entweder Rendite oder RSI)
    #    Sequenzlänge = window_size + 1 (10 Renditen + 1 RSI)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test .reshape((X_test .shape[0], X_test .shape[1], 1))

    # 7) LSTM-Modelldefinition
    model = Sequential([
        Input(shape=(window_size + 1, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=16,
              validation_split=0.1, verbose=0)

    # 8) Vorhersage der Log-Renditen
    y_pred = model.predict(X_test).flatten()
    rm_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_ret)

    # 9) Rückrechnung der Renditen in Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx       = i + window_size
        prev_price = test_df["Close"].iloc[idx-1]
        p_pred     = prev_price * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])
    rm_price = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_price)

    print(f"Fold {fold} – RMSE Log‑Renditen: {rm_ret:.4f}, RMSE Preise: {rm_price:.4f}")

    # letzter Fold zum Plotten merken
    if fold == n_splits:
        last_fold = {
            "index":    test_df.index[window_size:],
            "y_test":   y_test,
            "y_pred":   y_pred,
            "actuals":  actuals,
            "preds":    preds
        }
    fold += 1

# 10) Durchschnittliche RMSE ausgeben
print(f"\nDurchschnittlicher RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschnittlicher RMSE Aktienkurse: {np.mean(rmse_prices):.4f}")

# 11) Plots für den letzten Fold
if last_fold:
    idx = last_fold["index"]

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_test"],  label="Tatsächliche Log‑Renditen")
    plt.plot(idx, last_fold["y_pred"],  label="Vorhergesagte Log‑Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log‑Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log‑Rendite"); plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["actuals"], label="Tatsächlicher Kurs")
    plt.plot(idx, last_fold["preds"],   label="Vorhergesagter Kurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend()
    plt.show()
