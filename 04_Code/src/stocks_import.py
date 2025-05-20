import datetime
import yfinance as yf
import time

# Zeitraum: letzte 30 Tage
today = datetime.date.today()
start_date = today - datetime.timedelta(days=60)
end_date = today

# Aktienkurse (täglich)
tickers = ["NVDA", "GOOG", "MSFT"]
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    df.to_csv(f"../../03_Daten/raw_data/historical_stock_data_daily_{ticker}_last60d.csv")
    time.sleep(10)  # 10 Sekunden warten