import pandas as pd

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Berechnet den klassischen 14‑Perioden‑RSI mit min_periods,
    sodass erst ab Index = period Werte ausgegeben werden.
    """
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # hier min_periods=period setzen
    ema_up = up.ewm(com=period-1, adjust=False, min_periods=period).mean()
    ema_down = down.ewm(com=period-1, adjust=False, min_periods=period).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Pfad zur geflatteten CSV
#infile = "../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv"
#infile = "../../03_Daten/processed_data/historical_stock_data_weekly_GOOG_flat.csv"
infile = "../../03_Daten/processed_data/historical_stock_data_weekly_MSFT_flat.csv"
#outfile = "../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat_with_RSI.csv"
#outfile = "../../03_Daten/processed_data/historical_stock_data_weekly_GOOG_flat_with_RSI.csv"
outfile = "../../03_Daten/processed_data/historical_stock_data_weekly_MSFT_flat_with_RSI.csv"

# CSV einlesen
df = pd.read_csv(infile, parse_dates=["Date"], index_col="Date")

# RSI berechnen (erst ab der 14. Woche echte Werte)
df["RSI_14"] = compute_rsi(df, period=14)

# Kurzcheck auf NaNs in den ersten Reihen
print(df[["Close", "RSI_14"]].head(20))

# Neue CSV speichern
df.to_csv(outfile, index=True)
print(f"RSI hinzugefügt und gespeichert in {outfile}")
