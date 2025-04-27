import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL._imaging import display
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

tickers = ["NVDA", "GOOG", "MSFT"]
results_baseline = []

for ticker in tickers:
    df = pd.read_csv(f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
                     parse_dates=['Date'], index_col='Date').sort_index()

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # --- Naives Modell ---
    test["Naive_Forecast"] = test["Close"].shift(1)
    test.iloc[0, test.columns.get_loc("Naive_Forecast")] = train["Close"].iloc[-1]

    test_naive = test.dropna(subset=["Naive_Forecast"])
    y_true = test_naive["Close"]
    y_pred_naive = test_naive["Naive_Forecast"]

    rmse_naive = math.sqrt(mean_squared_error(y_true, y_pred_naive))
    mae_naive  = mean_absolute_error(y_true, y_pred_naive)
    mape_naive = np.mean(np.abs((y_true - y_pred_naive) / y_true)) * 100
    r2_naive   = r2_score(y_true, y_pred_naive)
    # Hit-Rate: Anteil der richtigen Richtung
    direction_true = np.sign(y_true.diff().dropna())
    direction_pred = np.sign(y_pred_naive.diff().dropna())
    hit_rate_naive = (direction_true == direction_pred).mean() * 100

    # --- ARIMA(1,1,1) ---
    arima = ARIMA(train["Close"], order=(1,1,1)).fit()
    forecast_arima = arima.forecast(steps=len(test))
    forecast_arima.index = test.index

    y_true_arima = test["Close"]
    y_pred_arima = forecast_arima

    rmse_arima = math.sqrt(mean_squared_error(y_true_arima, y_pred_arima))
    mae_arima  = mean_absolute_error(y_true_arima, y_pred_arima)
    mape_arima = np.mean(np.abs((y_true_arima - y_pred_arima) / y_true_arima)) * 100
    r2_arima   = r2_score(y_true_arima, y_pred_arima)
    direction_true_arima = np.sign(y_true_arima.diff().dropna())
    direction_pred_arima = np.sign(y_pred_arima.diff().dropna())
    hit_rate_arima = (direction_true_arima == direction_pred_arima).mean() * 100

    print(f"=== Basismodell für {ticker} ===")
    print(f"Naives Modell:    RMSE={rmse_naive:.4f}, MAE={mae_naive:.4f}, "
          f"MAPE={mape_naive:.2f}%, R²={r2_naive:.4f}, Hit-Rate={hit_rate_naive:.1f}%")
    print(f"ARIMA(1,1,1):     RMSE={rmse_arima:.4f}, MAE={mae_arima:.4f}, "
          f"MAPE={mape_arima:.2f}%, R²={r2_arima:.4f}, Hit-Rate={hit_rate_arima:.1f}%\n")

    results_baseline.append({
        "Ticker": ticker,
        "RMSE_Naive": rmse_naive,  "MAE_Naive": mae_naive,
        "MAPE_Naive": mape_naive,  "R2_Naive": r2_naive,
        "HitRate_Naive": hit_rate_naive,
        "RMSE_ARIMA": rmse_arima,  "MAE_ARIMA": mae_arima,
        "MAPE_ARIMA": mape_arima,  "R2_ARIMA": r2_arima,
        "HitRate_ARIMA": hit_rate_arima
    })

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    # Echte Kurse
    plt.plot(train.index, train["Close"],
             label="Train Close", color="tab:blue", linewidth=2.5, zorder=2)
    plt.plot(test.index, test["Close"],
             label="Test Close",  color="tab:blue", linewidth=2.5, zorder=2)
    # ARIMA
    plt.plot(forecast_arima.index, forecast_arima,
             label="ARIMA Forecast", color="tab:green",
             linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)
    # Naives Modell
    plt.plot(test_naive.index, test_naive["Naive_Forecast"],
             label="Naive Forecast", color="tab:red",
             linestyle=":", linewidth=2.5, marker="o", markersize=4, zorder=3)

    plt.title(f"{ticker} – Baseline-Modelle")
    plt.xlabel("Datum")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Zusammenfassung
df_summary = pd.DataFrame(results_baseline)
print("\n=== Zusammenfassung Basismodelle ===")
display(df_summary)
