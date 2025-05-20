import numpy as np
import pandas as pd
import math
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# 1) Daten laden und auf die letzten 30 Tage beschränken
ticker = "MSFT"
df = pd.read_csv(f"../../03_Daten/processed_data/historical_stock_data_daily_{ticker}_last60d_flat_with_RSI.csv", parse_dates=["Date"])
df = df.sort_values("Date").set_index("Date")
df = df.last("30D").copy()

# 2) Return und GARCH-Volatilität berechnen
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(subset=["Return", "RSI_14"], inplace=True)

# GARCH auf Return anwenden
scaled_ret = df["Return"] * 10
garch = arch_model(scaled_ret, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = garch.conditional_volatility / 10

# 3) Feature-Skalierung (wie im Training: StandardScaler)
scaler = StandardScaler()
df[["GARCH_vol", "RSI_14"]] = scaler.fit_transform(df[["GARCH_vol", "RSI_14"]])

# 4) Sequenzen erzeugen (wie im Training)
window_size = 10
def make_xy(df):
    X, y = [], []
    for i in range(window_size, len(df)):
        seq  = df["Return"].iloc[i - window_size:i].tolist()
        stat = df[["GARCH_vol", "RSI_14"]].iloc[i].tolist()
        X.append(seq + stat)
        y.append(df["Return"].iloc[i])
    return np.array(X), np.array(y)

X_test, y_test = make_xy(df)
X_test = X_test.reshape((-1, X_test.shape[1], 1))

# 5) Modell laden und Vorhersage durchführen
model = load_model(f"../../05_Modelle/garch_lstm_{ticker.lower()}_final_model.keras")
y_pred = model.predict(X_test).flatten()

# 6) RMSE berechnen
rmse_day  = math.sqrt(mean_squared_error(y_test[-1:], y_pred[-1:]))
rmse_week = math.sqrt(mean_squared_error(y_test[-7:], y_pred[-7:]))
rmse_full = math.sqrt(mean_squared_error(y_test, y_pred))

# 7) Ergebnis anzeigen
print("\nRMSE auf den letzten 30 Tagen:")
print(f"Letzter Tag    : {rmse_day:.4f}")
print(f"Letzte Woche   : {rmse_week:.4f}")
print(f"Letzte 30 Tage : {rmse_full:.4f}")

# 8) RMSE auf tatsächliche Preise berechnen
# Rekonstruiere Preise aus vorhergesagten Log-Renditen
pred_prices = []
real_prices = []

# Ausgangspunkt: Preis an Position window_size - 1
start_idx = window_size - 1
for i in range(len(y_pred)):
    p0 = df["Close"].iloc[start_idx + i]  # letzter bekannter Preis
    pred_price = p0 * np.exp(y_pred[i])
    real_price = df["Close"].iloc[start_idx + i + 1]
    pred_prices.append(pred_price)
    real_prices.append(real_price)

pred_prices = np.array(pred_prices)
real_prices = np.array(real_prices)

# Preis-RMSE berechnen
rmse_price_day  = math.sqrt(mean_squared_error(real_prices[-1:], pred_prices[-1:]))
rmse_price_week = math.sqrt(mean_squared_error(real_prices[-7:], pred_prices[-7:]))
rmse_price_full = math.sqrt(mean_squared_error(real_prices, pred_prices))

# Ausgabe
print("\nRMSE auf den tatsächlichen Aktienkursen:")
print(f"Letzter Tag    : {rmse_price_day:.4f}")
print(f"Letzte Woche   : {rmse_price_week:.4f}")
print(f"Letzte 30 Tage : {rmse_price_full:.4f}")
