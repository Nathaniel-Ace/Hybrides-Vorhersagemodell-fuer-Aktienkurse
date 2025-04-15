import pandas as pd
import matplotlib.pyplot as plt

# Liste der Ticker
tickers = ["NVDA", "GOOG", "MSFT"]

for ticker in tickers:
    # Laden der geflatteten Aktienkursdaten
    stock_file = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv"
    stock_df = pd.read_csv(stock_file, parse_dates=["Date"], index_col="Date")
    stock_df.sort_index(inplace=True)

    # Laden der Google Trends Daten für den jeweiligen Ticker
    trends_file = f"../../03_Daten/raw_data/google_trends_weekly_{ticker}.csv"
    trends_df = pd.read_csv(trends_file, parse_dates=["date"], index_col="date")
    trends_df.sort_index(inplace=True)

    # Aggregation der Google Trends Daten:
    # Berechnung des arithmetischen Mittelwerts über alle Spalten (Keywords)
    trends_df["Trend_Average"] = trends_df.mean(axis=1)

    # Glätten der Trend-Linie (z.B. gleitender Durchschnitt über 3 Zeitpunkte)
    trends_df["Trend_Smoothed"] = trends_df["Trend_Average"].rolling(window=3).mean()

    # Anzeigen der ersten Zeilen der aggregierten Daten
    print(f"\nErste Zeilen der aggregierten Google Trends Daten für {ticker}:")
    print(trends_df.head())

    # Plot 1: Schlusskurse und aggregierte Trend-Linie gemeinsam darstellen
    plt.figure(figsize=(12, 6))
    plt.plot(stock_df.index, stock_df["Close"], label=f"{ticker} Schlusskurs", color="blue")
    plt.plot(trends_df.index, trends_df["Trend_Average"], label="Trend Average", linestyle="--", color="orange",
             alpha=0.8)
    plt.plot(trends_df.index, trends_df["Trend_Smoothed"], label="Trend Smoothed", linestyle=":", color="green",
             alpha=0.8)
    plt.xlabel("Datum")
    plt.ylabel("Wert")
    plt.title(f"{ticker}: Schlusskurs und Aggregierte Google Trends")
    plt.legend()
    plt.show()

    # Plot 2: Separater Plot nur für die aggregierte Trend-Linie
    plt.figure(figsize=(12, 4))
    plt.plot(trends_df.index, trends_df["Trend_Average"], label="Trend Average", color="orange", linestyle="--")
    plt.plot(trends_df.index, trends_df["Trend_Smoothed"], label="Trend Smoothed", color="green", linestyle=":")
    plt.xlabel("Datum")
    plt.ylabel("Suchinteresse")
    plt.title(f"{ticker}: Aggregierte Google Trends Zeitreihe")
    plt.legend()
    plt.show()
