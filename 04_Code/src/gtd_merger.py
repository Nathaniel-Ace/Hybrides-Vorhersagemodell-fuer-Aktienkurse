import os
import pandas as pd

# Einstellungen
tickers = ["NVDA", "GOOG", "MSFT"]
periods = ["2015-2020", "2020-2023", "2023-2025"]
input_pattern = "../../03_Daten/raw_data/google_trends_weekly_{ticker}_{period}.csv"
output_dir = "../../03_Daten/processed_data"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    # Liste zum Sammeln der DataFrames
    dfs = []

    # 1) Die CSVs für die 3 Perioden einlesen
    for period in periods:
        fn = input_pattern.format(ticker=ticker, period=period)
        try:
            df = pd.read_csv(
                fn,
                parse_dates=["date"],
                index_col="date"
            )
            dfs.append(df)
            print(f"eingelesen: {fn} ({len(df)} Zeilen)")
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {fn}")

    if not dfs:
        print(f"Keine Daten für {ticker}, überspringe.")
        continue

    # 2) Aneinanderhängen
    merged = pd.concat(dfs)

    # 3) Nach Datum sortieren
    merged = merged.sort_index()

    # 4) Duplikate (gleicher Index) entfernen, ersten behalten
    merged = merged[~merged.index.duplicated(keep="first")]

    # 5) Abspeichern
    out_fn = os.path.join(
        output_dir,
        f"google_trends_weekly_{ticker}_2015-2025.csv"
    )
    merged.to_csv(out_fn)
    print(f"gespeichert: {out_fn} ({len(merged)} Zeilen)\n")
