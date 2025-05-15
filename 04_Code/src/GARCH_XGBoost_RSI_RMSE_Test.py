from joblib import load
from arch import arch_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math

# Parameter
ticker = "NVDA"
window_size = 10
model_path = f"../../05_Modelle/garch_xgboost_{ticker.lower()}_final_model.joblib"
csv_path = f"../../03_Daten/processed_data/historical_stock_data_daily_{ticker}_last60d_flat_with_RSI.csv"

# Modell laden
model = load(model_path)
print(f"✅ Modell geladen: {model_path}")

# Daten laden und vorbereiten
df = pd.read_csv(csv_path, parse_dates=["Date"])
df = df.sort_values("Date").set_index("Date")
df = df.loc[df.index >= df.index.max() - pd.Timedelta(days=30)].copy()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(subset=["Return", "RSI_14"], inplace=True)

# GARCH-Volatilität berechnen
ret_scaled = df["Return"] * 10
garch = arch_model(ret_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = garch.conditional_volatility / 10

# Feature-Vektoren erstellen
def make_xy_outsample(df, window_size):
    X, y = [], []
    for i in range(window_size, len(df)):
        seq = df["Return"].iloc[i - window_size:i].values
        vol = df["GARCH_vol"].iloc[i]
        rsi = df["RSI_14"].iloc[i]
        X.append(np.concatenate([seq, [vol, rsi]]))
        y.append(df["Return"].iloc[i])
    return np.array(X), np.array(y)

X_30, y_30 = make_xy_outsample(df, window_size)
y_pred = model.predict(X_30)

# RMSE Log-Rendite
rmse_ret_day  = math.sqrt(mean_squared_error(y_30[-1:], y_pred[-1:]))
rmse_ret_week = math.sqrt(mean_squared_error(y_30[-7:], y_pred[-7:]))
rmse_ret_full = math.sqrt(mean_squared_error(y_30, y_pred))
print("\n📈 RMSE auf Log-Renditen (Out-of-sample)")
print(f"Letzter Tag    : {rmse_ret_day:.4f}")
print(f"Letzte Woche   : {rmse_ret_week:.4f}")
print(f"Letzte 30 Tage : {rmse_ret_full:.4f}")

# RMSE Preisprognose
pred_prices, real_prices = [], []
start_idx = window_size - 1
for i in range(len(y_pred)):
    p0 = df["Close"].iloc[start_idx + i]
    p_pred = p0 * np.exp(y_pred[i])
    p_real = df["Close"].iloc[start_idx + i + 1]
    pred_prices.append(p_pred)
    real_prices.append(p_real)

rmse_price_day  = math.sqrt(mean_squared_error(real_prices[-1:], pred_prices[-1:]))
rmse_price_week = math.sqrt(mean_squared_error(real_prices[-7:], pred_prices[-7:]))
rmse_price_full = math.sqrt(mean_squared_error(real_prices, pred_prices))
print("\n💰 RMSE auf Preisen (Out-of-sample)")
print(f"Letzter Tag    : {rmse_price_day:.4f}")
print(f"Letzte Woche   : {rmse_price_week:.4f}")
print(f"Letzte 30 Tage : {rmse_price_full:.4f}")
