import pandas as pd

ticker = "MSFT"
gtd = "Microsoft"

# 1) Pfade anpassen
input_path = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025.csv"
output_path = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025_with_trends.csv"

# 2) CSV einlesen (Datum als Index)
df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)

# 3) Liste der GTD‑Spalten
gtd_cols = [f"{gtd} stock", f"sell {gtd} stock", f"buy {gtd} stock"]

# 4) Trend‑Durchschnitt und geglättete Trend‑Spalte berechnen
df["Trend_Average"] = df[gtd_cols].mean(axis=1)
df["Trend_Smoothed"] = df["Trend_Average"].rolling(window=3, min_periods=1).mean()

# 4b) Ursprüngliche GTD‑Spalten entfernen
df.drop(columns=gtd_cols, inplace=True)

# 5) Kurzer Blick aufs Ergebnis
print(df[["Trend_Average", "Trend_Smoothed"]].head(10))

# 6) In neue CSV speichern
df.to_csv(output_path)
print(f"Gespeichert: {output_path}")
