import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
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

    # 3) SARIMAX-Modell mit wöchentlicher Saisonperiode (52 Wochen)
    model = SARIMAX(
        train["Close"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)

    # 4) Forecast für den Test-Zeitraum
    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    # 5) RMSE-Berechnung
    rmse = math.sqrt(mean_squared_error(test["Close"], forecast))
    results.append({"Ticker": ticker, "RMSE_SARIMAX": rmse})
    print(f"{ticker} – SARIMAX RMSE: {rmse:.4f}")

    # 6) Plot: Train / Test / Forecast
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train["Close"], label="Train (Close)", color="gray")
    plt.plot(test.index, test["Close"], label="Test (Close)", color="black")
    plt.plot(forecast.index, forecast, label="SARIMAX Forecast", alpha=0.8)
    plt.title(f"{ticker}: SARIMAX(1,1,1)x(1,1,1,52) Forecast")
    plt.xlabel("Datum")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# 7) Zusammenfassung aller RMSE
df_results = pd.DataFrame(results)
print("\n=== SARIMAX-Ergebnisse ===")
print(df_results.to_string(index=False))
