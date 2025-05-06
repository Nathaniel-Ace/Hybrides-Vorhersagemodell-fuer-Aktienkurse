import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from xgboost import XGBRegressor

# 1) CSV laden und Daten vorbereiten
ticker = "NVDA"
df = pd.read_csv(
    f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
    parse_dates=["Date"], index_col="Date"
).sort_index()
data = df[["Close"]].copy()
values = data.values  # shape: (n_samples, 1)

# 2) Funktion zum Erstellen von Sequenzen (Windowing)
def create_sequences(dataset, window_size=10):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i : i + window_size, 0])
        y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10

# 3) TimeSeriesSplit einrichten
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# 4) Storage für Metriken und Kursverläufe
rmse_list, mae_list, mape_list, r2_list = [], [], [], []
hit_rate_list, sharpe_list = [], []
all_y_true, all_y_pred, all_dates = [], [], []

# 5) Loop über Folds
fold_num = 1
for train_idx, test_idx in tscv.split(values):
    # a) Train/Test-Split
    train_data = values[train_idx]
    test_data  = values[test_idx]

    # b) Sequenzen erstellen
    X_train, y_train = create_sequences(train_data, window_size)
    X_test,  y_test  = create_sequences(test_data,  window_size)

    print(f"Fold {fold_num}: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")

    # c) XGBoost definieren & trainieren
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # d) Vorhersage
    y_pred = xgb_model.predict(X_test)

    # Datums-Indizes für Plot (warum window_size? weil die ersten window_size Samples pro Fold entfallen)
    dates = df.index[test_idx][window_size:]
    all_dates.append(dates)

    # Speichern der echten und prognostizierten Kurse
    all_y_true.append(y_test)
    all_y_pred.append(y_pred)

    # e) Metriken berechnen
    rmse  = math.sqrt(mean_squared_error(y_test, y_pred))
    mae   = mean_absolute_error(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2    = r2_score(y_test, y_pred)

    # Ergebnisse sammeln
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape)
    r2_list.append(r2)

    # Konsolenausgabe pro Fold
    print(f"Fold {fold_num} – "
          f"RMSE={rmse:.4f}, "
          f"MAE={mae:.4f}, "
          f"MAPE={mape:.2f}%, "
          f"R²={r2:.4f}")
    fold_num += 1

# 6) Durchschnitt über alle Folds
print("=== Durchschnitt über alle Folds ===")
print(f"RMSE:       {np.mean(rmse_list):.4f}")
print(f"MAE:        {np.mean(mae_list):.4f}")
print(f"MAPE:       {np.mean(mape_list):.4f}%")
print(f"R²:         {np.mean(r2_list):.4f}")

# 7) Plot für alle 3 Folds
plt.figure(figsize=(12, 6))
for i in range(n_splits):
    plt.plot(all_dates[i], all_y_true[i],
             label=f"Real Prc Fold {i+1}")
    plt.plot(all_dates[i], all_y_pred[i],
             linestyle="--", label=f"Pred Prc Fold {i+1}")
plt.title(f"{ticker} – Aktienkurse alle {n_splits} Folds")
plt.xlabel("Datum")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
