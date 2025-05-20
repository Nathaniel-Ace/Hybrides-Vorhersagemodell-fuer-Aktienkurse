import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Ticker-Liste und Ergebnis-Speicher
tickers = ["NVDA", "GOOG", "MSFT"]
results_sarimax = []

for ticker in tickers:
    # Daten einlesen
    df = pd.read_csv(
        f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
        parse_dates=["Date"],
        index_col="Date"
    ).sort_index()

    # TimeSeriesSplit-Validierung
    tscv = TimeSeriesSplit(n_splits=3)
    fold = 1
    plt.figure(figsize=(12, 8))

    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test  = df.iloc[test_idx]

        model = SARIMAX(
            train['Close'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)

        # Forecast für Test
        forecast = fit.forecast(steps=len(test))
        forecast.index = test.index

        # Metriken berechnen
        y_true = test['Close']
        y_pred = forecast
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2   = r2_score(y_true, y_pred)

        results_sarimax.append({
            'Ticker': ticker,
            'Fold': fold,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

        # Plot Actual vs Forecast
        plt.plot(test.index, y_true, label=f'Fold {fold} Actual', linewidth=2)
        plt.plot(forecast.index, forecast, linestyle='--', label=f'Fold {fold} Forecast', linewidth=2)
        fold += 1

    plt.title(f"{ticker} SARIMAX(1,1,1)x(1,1,1,52) Forecasts across 3 Folds")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Zusammenfassung der Fold-Metriken
summary_df = pd.DataFrame(results_sarimax)
print("Einzel-Fold SARIMAX Metriken:")
print(summary_df)

# Durchschnittliche Metriken je Ticker
avg_metrics = summary_df.groupby('Ticker').agg(
    RMSE=('RMSE','mean'),
    MAE=('MAE','mean'),
    MAPE=('MAPE','mean'),
    R2=('R2','mean')
).reset_index()

print("\nDurchschnittliche SARIMAX Metriken je Ticker über 3 Folds:")
print(avg_metrics)
