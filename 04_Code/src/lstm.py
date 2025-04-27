import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# 1) Daten einlesen und vorbereiten
ticker = "NVDA"
df = pd.read_csv(
    f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
    parse_dates=["Date"], index_col="Date"
).sort_index()

data = df[["Close"]].copy()

# Skalierung in [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 2) Sequenzen erzeugen
def create_sequences(dataset, window_size=10):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i : i + window_size, 0])
        y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10

# 3) Cross-Validation setup
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Listen für Metriken
rmse_list, mae_list, mape_list, r2_list, hit_rate_list, sharpe_list = ([] for _ in range(6))

fold_num = 1
for train_idx, test_idx in tscv.split(scaled_data):
    # Trainings- und Testdaten splitten
    train_data = scaled_data[train_idx]
    test_data  = scaled_data[test_idx]

    # Sequenzen bauen
    X_train, y_train = create_sequences(train_data, window_size)
    X_test,  y_test  = create_sequences(test_data,  window_size)

    # Reshape für LSTM
    X_train = X_train.reshape((-1, window_size, 1))
    X_test  = X_test .reshape((-1, window_size, 1))

    # 4) LSTM-Modell definieren und kompilieren
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Einmalige @tf.function für Inferenz
    @tf.function
    def predict_fn(x):
        return model(x, training=False)

    # 5) Vorhersage & inverse Skalierung
    y_pred_scaled = predict_fn(tf.constant(X_test)).numpy().reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test_scaled).flatten()

    # 6) Metriken berechnen
    rmse  = math.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2    = r2_score(y_true, y_pred)

    # Hit-Rate (Richtungstreffer)
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    hit_rate = (dir_true == dir_pred).mean() * 100

    # Sharpe-Ratio auf prognostizierten Renditen
    pred_returns = np.diff(y_pred) / y_pred[:-1]
    sharpe = pred_returns.mean() / pred_returns.std() if pred_returns.std() != 0 else np.nan

    # Ergebnisse sammeln
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape)
    r2_list.append(r2)
    hit_rate_list.append(hit_rate)
    sharpe_list.append(sharpe)

    # Fold-Ergebnis ausgeben
    print(f"Fold {fold_num} – RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, "
          f"R²={r2:.4f}, Hit-Rate={hit_rate:.1f}%, Sharpe={sharpe:.4f}")
    fold_num += 1

# 7) Durchschnittliche Metriken
print("\n=== Durchschnitt aller Folds ===")
print(f"RMSE:       {np.mean(rmse_list):.4f}")
print(f"MAE:        {np.mean(mae_list):.4f}")
print(f"MAPE:       {np.mean(mape_list):.2f}%")
print(f"R²:         {np.mean(r2_list):.4f}")
print(f"Hit-Rate:   {np.mean(hit_rate_list):.1f}%")
print(f"Sharpe:     {np.nanmean(sharpe_list):.4f}")

# 8) Plot für den letzten Fold
plt.figure(figsize=(10,5))
plt.plot(y_true, label="Tatsächliche Close-Preise")
plt.plot(y_pred, label="LSTM Vorhersagen", alpha=0.7)
plt.title(f"{ticker} – LSTM vs. Realität (letzter Fold)")
plt.xlabel("Test Sample")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

