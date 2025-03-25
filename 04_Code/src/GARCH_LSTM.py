import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import math
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# 1) Daten einlesen und logarithmische Renditen berechnen
df = pd.read_csv("../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv",
                 parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
# Berechne logarithmische Renditen
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Train-Test-Split
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# 3) GARCH-Modell auf den Trainingsrenditen fitten
# Um den DataScaleWarning zu umgehen, skalieren wir die Renditen vor dem Fit mit 10.
train_returns_scaled = train_df["Return"] * 10
garch_model = arch_model(train_returns_scaled, mean='Zero', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
res = garch_model.fit(disp='off')
print(res.summary())

# Nach der Schätzung: Bedingte Volatilität auf Originalskala
train_df["GARCH_vol"] = res.conditional_volatility / 10

# 4) GARCH-Volatilitätsprognose für den Testbereich
forecast = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
test_vol = np.sqrt(forecast.variance.values[-1, :]) / 10
test_df["GARCH_vol"] = test_vol

# 5) Features erstellen: Für jeden Zeitpunkt ab window_size verwenden wir die letzten 'window_size' Renditen
# plus den GARCH-Volatilitätswert des Zielzeitpunkts als zusätzliches Feature.
def create_features_and_target(df, window_size=10):
    X, y = [], []
    returns = df["Return"].values
    vol = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        # Die letzten window_size Renditen als Sequenz...
        features = list(returns[i-window_size:i])
        # ... und zusätzlich der GARCH-Volatilitätswert zum Zeitpunkt i.
        features.append(vol[i])
        X.append(features)
        y.append(returns[i])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_features_and_target(train_df, window_size)
X_test, y_test = create_features_and_target(test_df, window_size)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# 6) Für LSTM: Reshape der Feature-Matrix in 3D (samples, timesteps, features)
# Hier entspricht die Sequenzlänge window_size+1, und wir haben 1 Feature pro Zeitschritt.
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 7) LSTM-Modell definieren
model = Sequential()
model.add(Input(shape=(window_size+1, 1)))  # window_size vergangene Renditen + 1 Volatilitätswert
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))  # Vorhersage: Logarithmischer Return
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 8) LSTM-Modell trainieren
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# 9) Vorhersage auf dem Testset
y_pred = model.predict(X_test).reshape(-1)
y_test = y_test.reshape(-1)
rmse_returns = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE auf logarithmische Renditen: {rmse_returns:.4f}")

# 10) Aus den vorhergesagten log-Renditen werden Aktienkurse rekonstruiert.
# Für jedes Test-Sample entspricht das Ziel r(t) dem logarithmischen Return zum Zeitpunkt t.
# Der vorhergesagte Aktienkurs P_pred(t) wird berechnet als:
# P_pred(t) = P(t-1) * exp( r_pred(t) )
predicted_prices = []
actual_prices = []
for i, r_pred in enumerate(y_pred):
    target_idx = i + window_size  # Der entsprechende Index im test_df, da wir 'window_size' Samples verloren haben
    prev_price = test_df["Close"].iloc[target_idx - 1]
    price_pred = prev_price * np.exp(r_pred)
    predicted_prices.append(price_pred)
    actual_prices.append(test_df["Close"].iloc[target_idx])
predicted_prices = np.array(predicted_prices)
actual_prices = np.array(actual_prices)

rmse_price = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"RMSE auf Aktienkurs-Basis: {rmse_price:.4f}")

# 11) Plot 1: Vergleich der tatsächlichen vs. vorhergesagten logarithmischen Renditen
plt.figure(figsize=(12,6))
plt.plot(test_df.index[window_size:], y_test, label="Tatsächliche Log-Renditen")
plt.plot(test_df.index[window_size:], y_pred, label="Vorhergesagte Log-Renditen", alpha=0.7)
plt.xlabel("Datum")
plt.ylabel("Logarithmische Rendite")
plt.title("Hybrid-Modell (GARCH + LSTM): Log-Renditen")
plt.legend()
plt.show()

# 12) Plot 2: Vergleich der tatsächlichen vs. vorhergesagten Aktienkurse
plt.figure(figsize=(12,6))
plt.plot(test_df.index[window_size:], actual_prices, label="Tatsächlicher Aktienkurs")
plt.plot(test_df.index[window_size:], predicted_prices, label="Vorhergesagter Aktienkurs", alpha=0.7)
plt.xlabel("Datum")
plt.ylabel("Aktienkurs (Close)")
plt.title("Hybrid-Modell (GARCH + LSTM): Aktienkurse")
plt.legend()
plt.show()
