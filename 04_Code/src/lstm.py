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
ticker = "MSFT"
df = pd.read_csv(
    f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat.csv",
    parse_dates=["Date"], index_col="Date"
).sort_index()
data = df[["Close"]].copy()

# 2) Skalierung in [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3) Funktion, um Sequenzen zu bauen
def create_sequences(dataset, window_size=10):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i : i + window_size, 0])
        y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10

# 4) TimeSeriesSplit
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

# Storage
rmse_list, mae_list, mape_list, r2_list, hit_rate_list, sharpe_list = ([] for _ in range(6))
all_y_true, all_y_pred, all_dates = [], [], []

# 5) Loop über Folds
fold_num = 1
for train_idx, test_idx in tscv.split(scaled_data):
    # Split in train/test
    train_data = scaled_data[train_idx]
    test_data  = scaled_data[test_idx]

    # Sequenzen
    X_train, y_train = create_sequences(train_data, window_size)
    X_test,  y_test  = create_sequences(test_data,  window_size)

    # Reshape für LSTM
    X_train = X_train.reshape((-1, window_size, 1))
    X_test  = X_test.reshape((-1, window_size, 1))

    # Modell definieren
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

    # @tf.function Inferenz
    @tf.function
    def predict_fn(x):
        return model(x, training=False)

    # Vorhersage & inverse Skalierung
    y_pred_scaled = predict_fn(tf.constant(X_test)).numpy().reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test_scaled).flatten()

    # Datums-Indizes für diesen Fold (erste window_size Zeitpunkte entfallen)
    dates = df.index[test_idx][window_size:]
    all_dates.append(dates)

    # Metriken berechnen
    rmse  = math.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2    = r2_score(y_true, y_pred)
    dir_true   = np.sign(np.diff(y_true))
    dir_pred   = np.sign(np.diff(y_pred))
    hit_rate   = (dir_true == dir_pred).mean() * 100
    pred_rets  = np.diff(y_pred) / y_pred[:-1]
    sharpe     = pred_rets.mean() / pred_rets.std() if pred_rets.std() != 0 else np.nan

    # Speichern
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape)
    r2_list.append(r2)
    hit_rate_list.append(hit_rate)
    sharpe_list.append(sharpe)
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

    # Konsolenausgabe
    print(f"Fold {fold_num} – RMSE={rmse:.4f}, MAE={mae:.4f}, "
          f"MAPE={mape:.2f}%, R²={r2:.4f}, Hit-Rate={hit_rate:.1f}%, Sharpe={sharpe:.4f}")
    fold_num += 1

# 6) Durchschnitt aller Folds
print("\n=== Durchschnitt aller Folds ===")
print(f"RMSE:     {np.mean(rmse_list):.4f}")
print(f"MAE:      {np.mean(mae_list):.4f}")
print(f"MAPE:     {np.mean(mape_list):.4f}%")
print(f"R²:       {np.mean(r2_list):.4f}")
print(f"Hit-Rate: {np.mean(hit_rate_list):.4f}%")
print(f"Sharpe:   {np.nanmean(sharpe_list):.4f}")

# 7) Ein einziger Plot für alle 3 Folds
plt.figure(figsize=(12, 6))
for i in range(n_splits):
    # Real
    plt.plot(all_dates[i], all_y_true[i], label=f"Real Prc Fold {i+1}")
    # Pred (gestrichelt)
    plt.plot(all_dates[i], all_y_pred[i], linestyle="--", label=f"Pred Prc Fold {i+1}")
plt.title(f"{ticker} – Aktienkurse alle {n_splits} Folds")
plt.xlabel("Datum")
plt.ylabel("Preis (Close)")
plt.legend()
plt.tight_layout()
plt.show()
