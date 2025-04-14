import time
from pytrends.request import TrendReq

start_date = "2015-01-01"
end_date = "2025-03-06"
timeframe_str = f"{start_date} {end_date}"

ticker_keywords = {
    #"NVDA": ["NVIDIA stock", "buy NVIDIA stock", "sell NVIDIA stock", "NVIDIA earnings"],
    #"GOOG": ["Google stock", "sell Google stock", "buy Google stock", "Alphabet stock", "Google earnings"],
    "MSFT": ["Microsoft stock", "buy Microsoft stock", "sell Microsoft stock", "MSFT earnings", "Microsoft earnings"]
}

# Erstellen einer pytrends-Instanz
pytrends = TrendReq(hl="en-US", tz=360)

# Für jede Aktie: Abrufen der Google Trends Daten für alle Suchbegriffe und Zusammenführen der Ergebnisse
for ticker, keywords in ticker_keywords.items():
    print(f"\nStarte Abruf der Google Trends Daten für {ticker}...")

    df_all = None  # DataFrame, in dem die Zeitreihen-Daten aller Keywords speichern
    for keyword in keywords:
        print(f"  Abrufe Keyword: {keyword}")
        kw_list = [keyword]
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe_str, geo='', gprop='')

        # Retry-Logik
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                trends_data = pytrends.interest_over_time()
                break  # Bei Erfolg
            except Exception as e:
                retry_count += 1
                print(
                    f"    Fehler beim Abruf von '{keyword}': {e}. Warte 60 Sekunden, Versuch {retry_count} von {max_retries}...")
                time.sleep(60)
        else:
            raise Exception(f"Mehrere Versuche für '{keyword}' fehlgeschlagen. Bitte überprüfen Sie Ihre Anfrage.")

        # Entfernen der 'isPartial'-Spalte, falls vorhanden
        if 'isPartial' in trends_data.columns:
            trends_data = trends_data.drop(columns=['isPartial'])

        # Umbenennen der Spalte in den Keyword-Namen
        trends_data = trends_data.rename(columns={keyword: f"{keyword}"})

        # Falls df_all noch nicht existiert: setze es auf den aktuellen DataFrame
        if df_all is None:
            df_all = trends_data[[f"{keyword}"]].copy()
        else:
            # Merge: Zusammenführen an der Datumsspalte
            df_all = df_all.merge(trends_data[[f"{keyword}"]], left_index=True, right_index=True, how="outer")

    # Resultate chronologisch sortieren
    df_all = df_all.sort_index()
    print(f"\nErste Zeilen der kombinierten Google Trends Daten für {ticker}:")
    print(df_all.head())

    # kombiniertes DataFrame als CSV speichern
    output_filename = f"../../03_Daten/raw_data/google_trends_weekly_{ticker}.csv"
    df_all.to_csv(output_filename)
    print(f"Google Trends Daten für {ticker} wurden in '{output_filename}' gespeichert.")
