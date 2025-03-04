import datetime
import time
import yfinance as yf
from pytrends.request import TrendReq

# Zeitraum festlegen: ab 01.01.2015 bis heute
today = datetime.date.today()
start_date = "2015-01-01"
end_date = today.strftime("%Y-%m-%d")

# -----------------------------------------------
# 1. Wöchentliche Aktienkurse mit yfinance abrufen
# -----------------------------------------------
tickers = ["NVDA", "GOOG", "MSFT"]

# Wöchentliche Daten (interval="1wk")
stock_data = yf.download(tickers, start=start_date, end=end_date, interval="1wk")
print("Wöchentliche Aktienkurse (head):")
print(stock_data.head())

stock_data.to_csv("../../03_Daten/raw_data/historical_stock_data_weekly.csv")

# -----------------------------------------------
# 2. Wöchentliche Google Trends Daten mit pytrends abrufen
# -----------------------------------------------

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["NVIDIA stock", "Google stock", "Microsoft stock"]
timeframe = "2015-01-01 2024-12-31"
pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')

# Retry-Logik
max_retries = 3
retry_count = 0
while retry_count < max_retries:
    try:
        trends_data = pytrends.interest_over_time()
        break  # Bei Erfolg Schleife verlassen
    except Exception as e:
        retry_count += 1
        print(f"Fehler aufgetreten: {e}. Warte 60 Sekunden, Versuch {retry_count} von {max_retries}")
        time.sleep(60)
else:
    raise Exception("Mehrere Versuche fehlgeschlagen. Bitte warte länger oder teile den Zeitraum auf.")

# Entferne 'isPartial', falls vorhanden
if 'isPartial' in trends_data.columns:
    trends_data = trends_data.drop(columns=['isPartial'])

print(trends_data.head())

trends_data.to_csv("../../03_Daten/raw_data/google_trends_weekly.csv")