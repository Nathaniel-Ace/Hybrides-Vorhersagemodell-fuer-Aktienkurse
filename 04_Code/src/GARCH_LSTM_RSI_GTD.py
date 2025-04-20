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
csv_path    = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2020-2025.csv"

# 1) Daten einlesen und Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) GTD‑Spalten ermitteln
#gtd_cols = [c for c in df.columns if "stock" in c.lower()]
# nur "NVIDIA stock" als GT‑Feature
gtd_cols = ["NVIDIA stock"]

# 3) Cross‑Validation Setup
tscv = TimeSeriesSplit(n_splits=n_splits)

# Listen für RMSEs und Fold‑Ergebnisse
rmse_returns = []
rmse_prices  = []
fold_results = []

def create_features_and_target(df, window_size=10):
    X, y = [], []
    for i in range(window_size, len(df)):
        ret_window = df["Return"].iloc[i-window_size:i].values.tolist()
        static_feats = [df["GARCH_vol"].iat[i], df["RSI_14"].iat[i]] \
                       + [df[col].iat[i] for col in gtd_cols]
        X.append(ret_window + static_feats)
        y.append(df["Return"].iat[i])
    return np.array(X), np.array(y)

# 4) Folds durchlaufen
fold = 1
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # GARCH fitten
    scaled = train_df["Return"] * 10
    garch = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    res   = garch.fit(disp="off")
    train_df["GARCH_vol"] = res.conditional_volatility / 10

    # GARCH‑Forecast für Test‑Set
    fc = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Features/Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    # LSTM‑Input formen
    X_train = X_train.reshape((-1, X_train.shape[1], 1))
    X_test  = X_test .reshape((-1, X_test.shape[1], 1))

    # LSTM‑Modell trainieren
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=16,
              validation_split=0.1, verbose=0)

    # Vorhersage Log‑Renditen
    y_pred = model.predict(X_test).flatten()
    rm_ret = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_ret)

    # Rückrechnung Kurse und RMSE
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        prev_price = test_df["Close"].iloc[i + window_size - 1]
        preds.append(prev_price * np.exp(r))
        actuals.append(test_df["Close"].iloc[i + window_size])
    rm_price = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_price)

    print(f"Fold {fold} – RMSE Log‑Renditen: {rm_ret:.4f}, RMSE Preise: {rm_price:.4f}")

    # Fold‑Ergebnisse speichern
    fold_results.append({
        "idx":      test_df.index[window_size:],
        "y_test":   y_test,
        "y_pred":   y_pred,
        "actuals":  actuals,
        "preds":    preds
    })

    fold += 1

# 5) Durchschnittliche RMSE ausgeben
print(f"\nDurchschnittlicher RMSE Log‑Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschnittlicher RMSE Aktienkurse:  {np.mean(rmse_prices):.4f}")

# 6) Gemeinsame Plots aller Folds

# -- Log‑Renditen
plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, start=1):
    plt.plot(fr["idx"], fr["y_test"],  label=f"Actual Fold {i}")
    plt.plot(fr["idx"], fr["y_pred"],  label=f"Pred Fold {i}", linestyle="--", alpha=0.7)
plt.title(f"{ticker} – Log‐Renditen über alle {n_splits} Folds")
plt.xlabel("Datum")
plt.ylabel("Log‑Rendite")
plt.legend()
plt.tight_layout()
plt.show()

# -- Aktienkurse
plt.figure(figsize=(12,5))
for i, fr in enumerate(fold_results, start=1):
    plt.plot(fr["idx"], fr["actuals"], label=f"Actual Fold {i}")
    plt.plot(fr["idx"], fr["preds"],   label=f"Pred Fold {i}", linestyle="--", alpha=0.7)
plt.title(f"{ticker} – Aktienkurse über alle {n_splits} Folds")
plt.xlabel("Datum")
plt.ylabel("Preis (Close)")
plt.legend()
plt.tight_layout()
plt.show()
