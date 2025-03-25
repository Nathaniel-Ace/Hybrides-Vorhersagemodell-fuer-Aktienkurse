import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# -----------------------------
# 1) Daten einlesen & Renditen berechnen
# -----------------------------
df = pd.read_csv("../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv",
                 parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
# Berechne logarithmische Renditen (log-Return)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# -----------------------------
# 2) TimeSeriesSplit einrichten
# -----------------------------
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Listen für RMSE-Ergebnisse
rmse_returns_list = []
rmse_prices_list = []

# Optional: Um die Ergebnisse pro Fold zu plotten, speichern wir die Daten eines bestimmten Folds
plot_fold = 3  # z. B. den letzten Fold plotten
fold_counter = 1

# Fenstergröße für LSTM-Features
window_size = 10


# Funktion: Features (Fenster aus vergangenen Renditen + aktueller GARCH_vol) und Ziel (nächster Return) erstellen
def create_features_and_target(df, window_size=10):
    X, y = [], []
    returns = df["Return"].values
    vol = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        features = list(returns[i - window_size:i])  # vergangene Renditen
        features.append(vol[i])  # GARCH-Volatilitätswert zum Zeitpunkt i
        X.append(features)
        y.append(returns[i])
    return np.array(X), np.array(y)


# Für jeden Fold: Train/Test-Split, GARCH fitten, LSTM trainieren und evaluieren
for train_index, test_index in tscv.split(df):
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()

    # -----------------------------
    # 3) GARCH-Modell auf Trainingsrenditen fitten
    # Um DataScaleWarning zu vermeiden, skaliere die Trainingsrenditen (Faktor 10)
    train_returns_scaled = train_df["Return"] * 10
    garch_model = arch_model(train_returns_scaled, mean='Zero', vol='GARCH', p=1, q=1,
                             dist='normal', rescale=False)
    res_garch = garch_model.fit(disp='off')

    # Nach dem Fit: Konditionale Volatilität auf Originalskala
    train_df["GARCH_vol"] = res_garch.conditional_volatility / 10

    # -----------------------------
    # 4) GARCH-Volatilitätsprognose für Testbereich
    forecast = res_garch.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_vol = np.sqrt(forecast.variance.values[-1, :]) / 10
    test_df["GARCH_vol"] = test_vol

    # -----------------------------
    # 5) Features und Ziel für LSTM erstellen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test, y_test = create_features_and_target(test_df, window_size)

    # Reshape für LSTM: [Samples, Time Steps, Features]
    # Hier entspricht die Sequenzlänge window_size+1 (window_size Renditen + 1 Volatilitätswert) und wir haben 1 Feature pro "Zeitpunkt"
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # -----------------------------
    # 6) LSTM-Modell definieren und trainieren
    model = Sequential()
    model.add(Input(shape=(window_size + 1, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)

    # -----------------------------
    # 7) Vorhersage der log-Renditen im Testbereich
    y_pred = model.predict(X_test).reshape(-1)
    rmse_returns = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns_list.append(rmse_returns)
    print(f"Fold {fold_counter} RMSE auf log-Renditen: {rmse_returns:.4f}")

    # -----------------------------
    # 8) Umrechnung der vorhergesagten log-Renditen in Aktienkurse
    # Für jedes Test-Sample: P_pred(t) = P(t-1) * exp(r_pred(t))
    predicted_prices = []
    actual_prices = []
    for i, r_pred in enumerate(y_pred):
        target_idx = i + window_size  # In test_df, da die ersten 'window_size' Samples zum Feature-Fenster gehören
        prev_price = test_df["Close"].iloc[target_idx - 1]
        price_pred = prev_price * np.exp(r_pred)
        predicted_prices.append(price_pred)
        actual_prices.append(test_df["Close"].iloc[target_idx])
    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)

    rmse_price = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
    rmse_prices_list.append(rmse_price)
    print(f"Fold {fold_counter} RMSE auf Aktienkurs-Basis: {rmse_price:.4f}")

    # -----------------------------
    # 9) Optional: Für einen bestimmten Fold, z. B. den letzten, Plots erstellen
    if fold_counter == plot_fold:
        # Plot Renditen
        plt.figure(figsize=(12, 6))
        plt.plot(test_df.index[window_size:], y_test, label="Tatsächliche log-Renditen")
        plt.plot(test_df.index[window_size:], y_pred, label="Vorhergesagte log-Renditen", alpha=0.7)
        plt.xlabel("Datum")
        plt.ylabel("Logarithmische Rendite")
        plt.title(f"Fold {fold_counter} - Log-Renditen")
        plt.legend()
        plt.show()

        # Plot Aktienkurse
        plt.figure(figsize=(12, 6))
        plt.plot(test_df.index[window_size:], actual_prices, label="Tatsächlicher Aktienkurs")
        plt.plot(test_df.index[window_size:], predicted_prices, label="Vorhergesagter Aktienkurs", alpha=0.7)
        plt.xlabel("Datum")
        plt.ylabel("Aktienkurs (Close)")
        plt.title(f"Fold {fold_counter} - Aktienkurse")
        plt.legend()
        plt.show()

    fold_counter += 1

# Durchschnittliche RMSE über alle Folds
avg_rmse_returns = np.mean(rmse_returns_list)
avg_rmse_prices = np.mean(rmse_prices_list)
print(f"\nDurchschnittlicher RMSE auf log-Renditen über {n_splits} Folds: {avg_rmse_returns:.4f}")
print(f"Durchschnittlicher RMSE auf Aktienkurs-Basis über {n_splits} Folds: {avg_rmse_prices:.4f}")
