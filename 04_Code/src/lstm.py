import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# 1) Daten einlesen und vorbereiten
# Passe den Pfad an deine CSV-Datei an
df = pd.read_csv("../../03_Daten/processed_data/historical_stock_data_weekly_NVDA_flat.csv",
                 parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)

# Wir nutzen die "Close"-Spalte
data = df[["Close"]].copy()

# Skalierung der Daten in den Bereich [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# 2) Funktion zum Erstellen von Sequenzen (Fenster) für das LSTM
def create_sequences(dataset, window_size=10):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i: i + window_size, 0])
        y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(y)


window_size = 10

# 3) TimeSeriesSplit für Cross-Validation
n_splits = 3  # Du kannst die Anzahl der Splits anpassen
tscv = TimeSeriesSplit(n_splits=n_splits)

rmse_list = []
fold_num = 1

# 4) Iteriere über die Splits
for train_index, test_index in tscv.split(scaled_data):
    # Erstelle Trainings- und Testdaten aus den Indizes
    train_data = scaled_data[train_index]
    test_data = scaled_data[test_index]

    # Erstelle Sequenzen
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    # Reshape: [Samples, Time Steps, Features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 5) LSTM-Modell definieren
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))  # explizit eine Input-Layer
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training (du kannst Epochs und Batchgröße anpassen)
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # 6) Vorhersagen auf dem Test-Set
    y_pred_scaled = model.predict(X_test)
    # Reshape für die inverse Transformation
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Inverse Transformation in Originalwerte
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    # RMSE berechnen
    mse = mean_squared_error(y_test_inv, y_pred)
    rmse = math.sqrt(mse)
    print(f"Fold {fold_num} RMSE: {rmse:.4f}")
    rmse_list.append(rmse)
    fold_num += 1

# Durchschnittlicher RMSE über alle Folds
avg_rmse = np.mean(rmse_list)
print(f"\nDurchschnittlicher RMSE über {n_splits} Folds: {avg_rmse:.4f}")

# Optional: Plot für den letzten Fold (Testdaten vs. Vorhersagen)
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Tatsächliche Werte")
plt.plot(y_pred, label="LSTM Vorhersagen", alpha=0.7)
plt.title("LSTM Vorhersage vs. Tatsächliche Werte (letzter Fold)")
plt.xlabel("Test Samples")
plt.ylabel("Close Price")
plt.legend()
plt.show()
