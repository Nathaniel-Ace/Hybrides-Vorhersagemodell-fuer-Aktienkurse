import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math

# 1) Daten einlesen und Renditen berechnen
df = pd.read_csv("../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv",
                 parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
# Berechne logarithmische Renditen
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Train/Test-Split (80% Training, 20% Test)
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# 3) GARCH-Modell auf Trainingsdaten fitten
# Um die Warnung zu vermeiden, werden die Renditen vor dem Fit mit 10 multipliziert.
train_returns_scaled = train_df["Return"] * 10
garch_model = arch_model(train_returns_scaled, mean='Zero', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
res = garch_model.fit(disp='off')
print(res.summary())

# Nach der Schätzung die bedingte Volatilität wieder auf Originalskala bringen
train_df["GARCH_vol"] = res.conditional_volatility / 10

# 4) GARCH-Volatilitätsprognose für den Testbereich
forecast = res.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
test_vol = np.sqrt(forecast.variance.values[-1, :]) / 10
test_df["GARCH_vol"] = test_vol

# 5) Feature-Erstellung für das ML-Modell: Für jeden Zeitpunkt ab window_size
#    werden die letzten 'window_size' Renditen plus der GARCH-Volatilitätswert (am Zielzeitpunkt) als Feature genutzt.
def create_features_and_target(df, window_size=10):
    X, y = [], []
    returns = df["Return"].values
    vol = df["GARCH_vol"].values
    for i in range(window_size, len(df)):
        features = list(returns[i-window_size:i])  # Vergangene Renditen
        features.append(vol[i])                    # GARCH-Volatilität zum Zeitpunkt i
        X.append(features)
        y.append(returns[i])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_features_and_target(train_df, window_size)
X_test, y_test = create_features_and_target(test_df, window_size)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# 6) XGBoost-Modell trainieren, um den nächsten logarithmischen Return vorherzusagen
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE auf Renditen: {rmse:.4f}")

# 7) Vorhergesagte Renditen in Aktienkurse umrechnen
# Für jedes Sample in X_test entspricht y_test[i] dem logarithmischen Return an Tag t,
# und der zugehörige Zeitpunkt in test_df ist test_df.index[i + window_size].
# Um den vorhergesagten Kurs zu berechnen, nehmen wir den tatsächlichen Kurs des vorherigen Zeitpunkts und multiplizieren ihn mit exp(predicted return).

predicted_prices = []
actual_prices = []

# Für jedes Sample im Testbereich:
for i, r_pred in enumerate(y_pred):
    # Der Index im Test-DataFrame, der zum Target gehört:
    target_idx = i + window_size
    # Der vorherige reale Kurs (C_{t-1}) ist:
    prev_price = test_df["Close"].iloc[target_idx - 1]
    # Vorhergesagter Preis = vorheriger Preis * exp(vorhergesagter logarithmischer Return)
    price_pred = prev_price * np.exp(r_pred)
    predicted_prices.append(price_pred)
    # Tatsächlicher Kurs zum Zeitpunkt t:
    actual_prices.append(test_df["Close"].iloc[target_idx])

predicted_prices = np.array(predicted_prices)
actual_prices = np.array(actual_prices)

# 8) Plot 1: Vergleich der tatsächlichen vs. vorhergesagten logarithmischen Renditen
plt.figure(figsize=(12,6))
plt.plot(test_df.index[window_size:], y_test, label="Tatsächliche Log-Renditen")
plt.plot(test_df.index[window_size:], y_pred, label="Vorhergesagte Log-Renditen", alpha=0.7)
plt.xlabel("Datum")
plt.ylabel("Logarithmische Rendite")
plt.title("Vergleich: Tatsächliche vs. Vorhergesagte Log-Renditen")
plt.legend()
plt.show()

# 9) Plot 2: Vergleich der tatsächlichen vs. vorhergesagten Aktienkurse
plt.figure(figsize=(12,6))
plt.plot(test_df.index[window_size:], actual_prices, label="Tatsächlicher Aktienkurs")
plt.plot(test_df.index[window_size:], predicted_prices, label="Vorhergesagter Aktienkurs", alpha=0.7)
plt.xlabel("Datum")
plt.ylabel("Aktienkurs (Close)")
plt.title("Vergleich: Tatsächlicher vs. Vorhergesagter Aktienkurs")
plt.legend()
plt.show()
