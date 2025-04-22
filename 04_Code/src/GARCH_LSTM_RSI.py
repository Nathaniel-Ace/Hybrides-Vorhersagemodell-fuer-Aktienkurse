import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

# Einstellungen
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Log‑Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Cross‑Validation Setup
tscv = TimeSeriesSplit(n_splits=n_splits)
rmse_returns = []
rmse_prices  = []

# 3) Feature‑Creator: Rolling‑Window der letzten `window_size` Renditen +
#    zum Zeitpunkt i: RSI_14 und GARCH_vol
def create_features_and_target(df, window_size=10):
    X, y = [], []
    for i in range(window_size, len(df)):
        # a) Rolling‑Window der letzten Renditen
        ret_seq = df["Return"].iloc[i-window_size:i].tolist()
        # b) statische Features zum Zeitpunkt i
        rsi       = df["RSI_14"].iat[i]
        garch_vol = df["GARCH_vol"].iat[i]
        features  = ret_seq + [rsi, garch_vol]
        X.append(features)
        y.append(df["Return"].iat[i])
    return np.array(X), np.array(y)

# 4) Schleife über die Folds
last_fold = None
fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 5) GARCH auf Trainingsrenditen fitten (skaliert um Faktor 10)
    scaled = train_df["Return"] * 10
    garch  = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False)
    res    = garch.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # 6) GARCH‑Forecast für den Testbereich
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # 7) Features und Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    print(f"Fold {fold}: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")

    # 8) Reshape für LSTM ([Samples, Timesteps, 1 Feature pro Schritt])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test .reshape((X_test .shape[0], X_test .shape[1], 1))

    # 9) LSTM‑Modell definieren und trainieren
    model = Sequential([
        Input(shape=(window_size + 2, 1)),  # +2 für RSI und GARCH_vol
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=16,
              validation_split=0.1, verbose=0)

    # 10) Vorhersage der Log‑Renditen
    y_pred = model.predict(X_test).flatten()
    rm_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_ret)

    # 11) Rückrechnung der Renditen in Kurse und RMSE
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx        = i + window_size
        prev_price = test_df["Close"].iloc[idx-1]
        p_pred     = prev_price * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])
    rm_price = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_price)

    print(f"Fold {fold} – RMSE Log‑Renditen: {rm_ret:.4f}, RMSE Preise: {rm_price:.4f}")

    # 12) Daten für letzten Fold merken
    if fold == n_splits:
        last_fold = {
            "index":   test_df.index[window_size:],
            "y_test":  y_test,
            "y_pred":  y_pred,
            "actuals": actuals,
            "preds":   preds
        }
    fold += 1

# 13) Durchschnittliche RMSE ausgeben
print(f"\nDurchschn. RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschn. RMSE Aktienkurse:  {np.mean(rmse_prices):.4f}")

# 14) Plots für den letzten Fold
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
