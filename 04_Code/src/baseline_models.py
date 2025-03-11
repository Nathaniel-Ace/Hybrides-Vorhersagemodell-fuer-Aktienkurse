import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

# Liste der Ticker und zugehörige Dateipfade für die geflatteten CSVs
tickers = ["NVDA", "GOOG", "MSFT"]
stock_data_dict = {}

for ticker in tickers:
    file_path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv"
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    stock_data_dict[ticker] = df
    print(f"{ticker} Aktienkurse (head):")
    print(df.head())

results_baseline = []

for ticker, df in stock_data_dict.items():
    print(f"\n=== Basismodell für {ticker} ===")

    # 1) Sortiere sicherheitshalber nach Datum
    df = df.sort_index()

    # 2) Lege einen Zeitreihen-Train/Test-Split fest
    #    Beispiel: die ersten 80% Training, letzte 20% Test
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    # ------------------------------
    # A) Naives Modell (Persistence)
    # ------------------------------
    # Prognose = letzter bekannter Close-Wert (aus t-1)
    # Wir verschieben die Spalte "Close" um 1 nach unten
    test["Naive_Forecast"] = test["Close"].shift(1)

    # Allerdings stammt der t-1-Wert aus dem Test-Set;
    # um streng "Zukunft" auszuschließen, kopieren wir den letzten Wert aus dem Training:
    # test.iloc[0].Naive_Forecast = train["Close"].iloc[-1]
    #
    # (Bei wöchentlichen Daten ist dieser Unterschied minimal,
    #  aber der Vollständigkeit halber wäre es korrekt,
    #  den letzten Wert aus train als Forecast für den ersten Test-Zeitpunkt zu nehmen.)

    # Fehlermaße
    # Wir entfernen Zeilen mit NaN (die erste Zeile im Test hat keinen shift-Wert)
    test_naive = test.dropna(subset=["Naive_Forecast"])
    mse_naive = mean_squared_error(test_naive["Close"], test_naive["Naive_Forecast"])
    rmse_naive = math.sqrt(mse_naive)

    print(f"Naives Modell: RMSE = {rmse_naive:.4f}")

    # -------------------------
    # B) Einfaches ARIMA-Modell
    # -------------------------
    # ARIMA(1,1,1) als Beispiel
    # Statsmodels erfordert, dass wir nur die Serie (Close) übergeben
    # d=1 bedeutet Differenzierung, um stationär zu werden
    model = ARIMA(train["Close"], order=(1, 1, 1))
    arima_results = model.fit()

    # Vorhersage für den gesamten Testzeitraum
    forecast_arima = arima_results.forecast(steps=len(test))
    forecast_arima.index = test.index  # Gleiche Indizes wie Test-Set

    # Fehlermaße
    mse_arima = mean_squared_error(test["Close"], forecast_arima)
    rmse_arima = math.sqrt(mse_arima)

    print(f"ARIMA(1,1,1): RMSE = {rmse_arima:.4f}")

    # Speichere Ergebnisse
    results_baseline.append({
        "Ticker": ticker,
        "RMSE_Naive": rmse_naive,
        "RMSE_ARIMA": rmse_arima
    })

    # -------------------------
    # Plot: Vergleich Prognose vs. Real
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train["Close"], label="Train (Close)")
    plt.plot(test.index, test["Close"], label="Test (Close)")
    plt.plot(forecast_arima.index, forecast_arima, label="ARIMA Forecast", alpha=0.7)
    plt.plot(test_naive.index, test_naive["Naive_Forecast"], label="Naive Forecast", alpha=0.7)
    plt.title(f"{ticker} - Vergleich Baseline-Modelle")
    plt.xlabel("Datum")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# Zusammenfassung drucken
df_results = pd.DataFrame(results_baseline)
print("\n=== Zusammenfassung Basismodelle ===")
display(df_results)
