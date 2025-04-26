import pandas as pd

# csv_path = "../../03_Daten/raw_data/historical_stock_data_weekly_NVDA.csv"
csv_path = "../../03_Daten/raw_data/historical_stock_data_weekly_GOOG.csv"
# csv_path = "../../03_Daten/raw_data/historical_stock_data_weekly_MSFT.csv"
# output_path = "../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv"
output_path = "../../03_Daten/processed_data/historical_stock_data_weekly_GOOG_flat.csv"
# output_path = "../../03_Daten/processed_data/historical_stock_data_weekly_MSFT_flat.csv"

# 1) Einlesen mit Angabe, dass 2 Headerzeilen vorhanden sind und der Index die "Date"-Spalte ist
df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

print("Vor Flatten:\n", df.head())
print("\nMultiIndex Columns:\n", df.columns)

# 2) Entferne die Ticker-Ebene (Level 1), sodass nur die Spaltennamen übrig bleiben
df.columns = df.columns.droplevel(1)
print("\nSpalten nach droplevel(1):\n", df.columns)

# 3) Falls das Datum aktuell als Index vorliegt, diesen zurück in eine eigene Spalte holen
df.reset_index(inplace=True)
# Falls der Indexname nicht bereits "Date" ist, kann man ihn umbenennen
if df.columns[0] != "Date":
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# 4) Wähle explizit die gewünschten Spalten in der gewünschten Reihenfolge
df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]

# 5) Datum in ein Datetime-Format umwandeln (optional, aber empfohlen)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 6) Das geflattete DataFrame als neue CSV speichern
df.to_csv(output_path, index=False)

print("\nNach Flatten:\n", df.head())
print(f"\nFlattened CSV gespeichert unter: {output_path}")
