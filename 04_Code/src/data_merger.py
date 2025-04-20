import pandas as pd
from pandas.tseries.offsets import DateOffset

# Liste der Ticker, für die die Daten zusammengeführt werden sollen
tickers = ["NVDA", "GOOG", "MSFT"]

for ticker in tickers:
    # 1) Aktienkurs‑ & RSI‑Daten einlesen (wöchentlich)
    stock_path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"
    stock_df = (
        pd.read_csv(stock_path, parse_dates=["Date"], index_col="Date")
          .sort_index()
    )

    # 2) Google Trends‑Daten einlesen (wöchentlich)
    trends_path = f"../../03_Daten/raw_data/google_trends_weekly_{ticker}.csv"
    trends_df = (
        pd.read_csv(trends_path, parse_dates=["Date"], index_col="Date")
          .sort_index()
    )

    # 3) Zeitraum auf die letzten 5 Jahre begrenzen
    #    wir nehmen das jeweils früher endende Datum beider Datensätze als Obergrenze
    end_date   = min(stock_df.index.max(), trends_df.index.max())
    start_date = end_date - DateOffset(years=5)
    stock_df   = stock_df.loc[start_date:end_date]
    trends_df  = trends_df.loc[start_date:end_date]

    # 4) Beide Indexe auf Wochenperioden (Mo–So) mappen
    stock_df.index  = stock_df.index.to_period("W")
    trends_df.index = trends_df.index.to_period("W")

    # 5) Inner Join auf der Period‑Index‑Ebene
    merged = stock_df.join(
        trends_df,
        how="inner",
        lsuffix="",
        rsuffix="_trend"
    )

    # 6) PeriodIndex zurück in Timestamps (Anfang der Woche = Montag)
    merged.index = merged.index.to_timestamp()

    # 7) Speichern
    out_path = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2020-2025.csv"
    merged.to_csv(out_path)
    print(f"{ticker}: Merged ({merged.shape[0]} Wochen) → {out_path}")
