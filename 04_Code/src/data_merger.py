import pandas as pd

# Liste der Ticker, für die die Daten zusammengeführt werden sollen
tickers = ["NVDA", "GOOG", "MSFT"]

for ticker in tickers:
    # 1) Aktienkurs- & RSI-Daten einlesen (wöchentlich), sortiert nach Datum
    stock_path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"
    stock_df = (
        pd.read_csv(stock_path, parse_dates=["Date"], index_col="Date")
          .sort_index()
    )

    # 2) Google Trends-Daten einlesen (wöchentlich), sortiert
    trends_path = f"../../03_Daten/processed_data/google_trends_weekly_{ticker}_2015-2025.csv"
    trends_df = (
        pd.read_csv(trends_path, parse_dates=["date"], index_col="date")
          .sort_index()
    )

    # 3) Gemeinsamen Zeithorizont bestimmen (ca. 10 Jahre)
    start_date = max(stock_df.index.min(), trends_df.index.min())
    end_date   = min(stock_df.index.max(), trends_df.index.max())
    stock_df   = stock_df.loc[start_date:end_date]
    trends_df  = trends_df.loc[start_date:end_date]

    # 4) Auf Wochenperioden abbilden (PeriodIndex)
    stock_df.index  = stock_df.index.to_period("W")
    trends_df.index = trends_df.index.to_period("W")

    # 5) Inner Join auf Wochenniveau
    merged = stock_df.join(
        trends_df,
        how="inner",
        lsuffix="",        # ggf. Suffixe anpassen, falls Spalten kollidieren
        rsuffix="_trend"
    )

    # 6) PeriodIndex zurück in Timestamps (erster Tag der Periode, i.d.R. Montag)
    merged.index = merged.index.to_timestamp()

    # 7) Abspeichern
    out_path = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025.csv"
    merged.to_csv(out_path)
    print(f"{ticker}: Merged über {merged.shape[0]} Wochen → {out_path}")
