import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

tickers = ["NVDA", "GOOG", "MSFT"]
results_arima = []

for ticker in tickers:
    df = pd.read_csv(
        f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
        parse_dates=['Date'], index_col='Date'
    ).sort_index()

    tscv = TimeSeriesSplit(n_splits=3)
    fold = 1
    plt.figure(figsize=(12, 8))

    for train_index, test_index in tscv.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        model = ARIMA(train['Close'], order=(1,1,1)).fit()
        forecast = model.forecast(steps=len(test))
        forecast.index = test.index

        # metrics
        y_true = test['Close']
        y_pred = forecast
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2   = r2_score(y_true, y_pred)

        results_arima.append({
            'Ticker': ticker,
            'Fold': fold,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

        plt.plot(test.index, test['Close'], label=f'Fold {fold} Actual', linewidth=2)
        plt.plot(forecast.index, forecast, linestyle='--', label=f'Fold {fold} Forecast', linewidth=2)
        fold += 1

    plt.title(f"{ticker} ARIMA(1,1,1) Forecasts across 3 Folds")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Zusammenfassung der Einzel-Fold-Metriken
summary_df = pd.DataFrame(results_arima)
print("Einzel-Fold Metriken:")
print(summary_df)

# Durchschnittliche Metriken über alle 3 Folds je Ticker
avg_metrics = summary_df.groupby('Ticker').agg(
    RMSE=('RMSE','mean'),
    MAE=('MAE','mean'),
    MAPE=('MAPE','mean'),
    R2=('R2','mean')
).reset_index()

print("\nDurchschnittliche Metriken je Ticker über 3 Folds:")
print(avg_metrics)
