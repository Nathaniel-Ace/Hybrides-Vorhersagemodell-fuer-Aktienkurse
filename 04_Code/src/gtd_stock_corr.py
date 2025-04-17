import pandas as pd
import matplotlib.pyplot as plt

tickers = ["NVDA", "GOOG", "MSFT"]

for ticker in tickers:
    # 1) Aktiendaten (wöchentlich)
    stock_file = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv"
    stock_df = pd.read_csv(stock_file, parse_dates=["Date"], index_col="Date").sort_index()

    # Ermittel den Wochentag für das Resampling
    first_wd = stock_df.index[0].dayofweek
    weekday_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
    freq = f"W-{weekday_map[first_wd]}"  # z.B. "W-THU"

    # 2) Monatliche Trends (monatsweise)
    trends_file = f"../../03_Daten/raw_data/google_trends_weekly_{ticker}.csv"
    trends_df = pd.read_csv(trends_file, parse_dates=["date"], index_col="date").sort_index()

    # 3) Auf Wochenbasis resamplen & interpolieren
    trends_weekly = (
        trends_df
        .resample(freq)  # z.B. "W-THU"
        .interpolate(method="linear")
        .loc[stock_df.index.min(): stock_df.index.max()]
    )

    # 4) Jetzt erst aggregieren
    trends_weekly["Trend_Average"] = trends_weekly.mean(axis=1)
    trends_weekly["Trend_Smoothed"] = trends_weekly["Trend_Average"].rolling(3).mean()

    # 5) Merge und Plot
    merged = stock_df.join(trends_weekly[["Trend_Average", "Trend_Smoothed"]], how="inner")

    plt.figure(figsize=(12, 6))
    plt.plot(merged.index, merged["Close"], label=f"{ticker} Schlusskurs")
    plt.plot(merged.index, merged["Trend_Average"], "--", label="Trend Average")
    plt.plot(merged.index, merged["Trend_Smoothed"], ":", label="Trend Smoothed")
    plt.title(f"{ticker}: Wöchentliche Kurse vs. Trends")
    plt.xlabel("Datum")
    plt.legend()
    plt.show()
