import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Liste der Ticker und Pfade zu den wöchentlichen CSVs
tickers = ["NVDA", "GOOG", "MSFT"]
results = []

for ticker in tickers:
    # 1) Daten einlesen und sortieren
    path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()

    # 2) Train/Test-Split (80% / 20%)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    # 3) SARIMAX(1,1,1)x(1,1,1,52)
    model = SARIMAX(
        train["Close"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)

    # 4) Forecast für Test-Zeitraum
    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    # 5) Basis-Metriken
    actual = test["Close"]
    rmse  = math.sqrt(mean_squared_error(actual, forecast))
    mae   = mean_absolute_error(actual, forecast)
    mape  = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2    = r2_score(actual, forecast)

    # 6) Sharpe-Ratio der prognostizierten wöchentlichen Renditen
    forecast_ret = forecast.pct_change().dropna()
    sharpe = forecast_ret.mean() / forecast_ret.std()

    # in Liste speichern
    results.append({
        "Ticker": ticker,
        "RMSE":   rmse,
        "MAE":    mae,
        "MAPE_%": mape,
        "R2":     r2,
        "Sharpe": sharpe
    })

    # Ausgabe
    print(f"{ticker} – SARIMAX Metriken:")
    print(f"  RMSE       = {rmse:.4f}")
    print(f"  MAE        = {mae:.4f}")
    print(f"  MAPE       = {mape:.2f}%")
    print(f"  R²         = {r2:.4f}")
    print(f"  Sharpe     = {sharpe:.4f}\n")

    # 7) Plot: Train / Test / Forecast
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train["Close"], label="Train", color="gray")
    plt.plot(test.index, actual,           label="Test",  color="black")
    plt.plot(forecast.index, forecast,     label="Forecast", alpha=0.8)
    plt.title(f"{ticker}: SARIMAX(1,1,1)x(1,1,1,52) — Forecast vs. Actual")
    plt.xlabel("Datum")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 8) Zusammenfassung aller Metriken
df_results = pd.DataFrame(results)
print("\n=== Zusammenfassung SARIMAX Metriken ===")
print(df_results.to_string(index=False))
