import pandas as pd

# Liste der Ticker, für die die Daten zusammengeführt werden sollen
tickers = ["NVDA", "GOOG", "MSFT"]

for ticker in tickers:
    # 1) Aktienkurs‑ und RSI‑Daten einlesen (wöchentlich)
    stock_path = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"
    stock_df = pd.read_csv(stock_path, parse_dates=["Date"], index_col="Date")
    stock_df.sort_index(inplace=True)

    # 2) Ermittlung des Wochentags (0=Montag ... 6=Sonntag)
    # Verwende .dayofweek (int), nicht .weekday (Methode)
    first_weekday = stock_df.index[0].dayofweek
    weekday_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
    resample_freq = f"W-{weekday_map[first_weekday]}"
    print(f"{ticker}: Aktiendaten indexieren auf Wochentag {weekday_map[first_weekday]} → Resample '{resample_freq}'")

    # 3) Monatliche Google Trends‑Daten einlesen
    trends_path = f"../../03_Daten/raw_data/google_trends_weekly_{ticker}.csv"
    monthly_trends = pd.read_csv(trends_path, parse_dates=["date"], index_col="date")
    monthly_trends.sort_index(inplace=True)

    # 4) Resample auf den gleichen Wochentag wie die Aktiendaten und linear interpolieren
    weekly_trends = (
        monthly_trends
        .resample(resample_freq)      # z. B. 'W-THU'
        .interpolate(method="linear") # lineare Interpolation zwischen Monatswerten
        .loc[stock_df.index.min():    # Zuschneiden auf exakt denselben Zeitraum
             stock_df.index.max()]
    )

    # 5) Merge der Aktien- und Trends‑Daten (inner join)
    merged = stock_df.join(weekly_trends, how="inner")

    # 6) Kurzer Kopfzeilen-Check
    print(f"\n{ticker} – merged head:")
    print(merged.head(), "\n")

    # 7) Zusammengeführtes DataFrame speichern
    out_path = f"../../03_Daten/processed_data/merged_weekly_{ticker}.csv"
    merged.to_csv(out_path)
    print(f"{ticker}: Fertig – '{out_path}' wurde erzeugt.\n")
