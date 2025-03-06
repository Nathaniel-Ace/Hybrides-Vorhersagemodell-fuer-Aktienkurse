import pandas as pd
import matplotlib.pyplot as plt

# Liste der Ticker und zugehörige Dateipfade für die geflatteten CSVs
tickers = ["NVDA", "GOOG", "MSFT"]
stock_data_dict = {}

for ticker in tickers:
    file_path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv"
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    stock_data_dict[ticker] = df
    print(f"{ticker} Aktienkurse (head):")
    print(df.head())

# Google Trends CSV einlesen (bleibt unverändert, da hier keine Anpassung nötig ist)
trends_data = pd.read_csv("../../03_Daten/raw_data/google_trends_weekly.csv", parse_dates=['date'], index_col='date')
print("\nGoogle Trends (head):")
print(trends_data.head())

# Für jeden Ticker: Daten zusammenführen (inner join anhand des Datums) und visualisieren
for ticker in tickers:
    merged_data = stock_data_dict[ticker].merge(trends_data, left_index=True, right_index=True, how='inner')
    print(f"\nZusammengeführte Daten für {ticker} (head):")
    print(merged_data.head())

    plt.figure(figsize=(10, 5))
    # Plot: Schlusskurs aus den Aktienkurs-Daten
    plt.plot(merged_data.index, merged_data['Close'], label=f'{ticker} Schlusskurs')

    # Auswahl der passenden Google Trends-Spalte
    if ticker == "NVDA":
        trends_col = "NVIDIA stock"
    elif ticker == "GOOG":
        trends_col = "Google stock"
    elif ticker == "MSFT":
        trends_col = "Microsoft stock"
    else:
        trends_col = None

    # Falls die Trends-Spalte vorhanden ist, auch diese plotten
    if trends_col and trends_col in merged_data.columns:
        plt.plot(merged_data.index, merged_data[trends_col], label=f'Google Trends: {trends_col}', alpha=0.7)

    plt.xlabel('Datum')
    plt.ylabel('Wert')
    plt.title(f'{ticker} Aktienkurs vs. Google Trends')
    plt.legend()
    plt.show()
