import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import math

# XGBoost
from xgboost import XGBRegressor

# 1) CSV laden und Daten vorbereiten
df = pd.read_csv("../../03_Daten/processed_data/historical_stock_data_weekly_GOOG_flat.csv",
                 parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)

# Wir verwenden nur die Spalte "Close"
data = df[["Close"]].copy()


# 2) Funktion zum Erstellen von Sequenzen (Windowing)
def create_sequences(dataset, window_size=10):
    """
    dataset: 2D-Array oder DataFrame (hier nur 1 Spalte, "Close")
    window_size: Anzahl der Zeitschritte, die als Input-Feature genutzt werden
    Gibt zurück: X (Features), y (Zielwerte)
    X wird 2D (Samples x window_size), y wird 1D (Samples)
    """
    X, y = [], []
    for i in range(len(dataset) - window_size):
        # Input: letzte 'window_size' Werte
        seq_x = dataset[i: i + window_size, 0]
        # Ziel: Wert am nächsten Zeitpunkt
        seq_y = dataset[i + window_size, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Wir wandeln DataFrame -> NumPy
values = data.values  # shape: (len, 1)

# 3) TimeSeriesSplit einrichten
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

rmse_list = []
fold_num = 1

for train_index, test_index in tscv.split(values):
    # Daten in Train/Test aufteilen
    train_data = values[train_index]
    test_data = values[test_index]

    # Sequenzen erstellen
    window_size = 10
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    # X ist 2D (Samples x window_size), y ist 1D
    print(
        f"Fold {fold_num}: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

    # 4) XGBoost-Modell definieren
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    # Training
    xgb_model.fit(X_train, y_train, verbose=False)

    # Vorhersage
    y_pred = xgb_model.predict(X_test)

    # RMSE berechnen
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    rmse_list.append(rmse)
    print(f"Fold {fold_num} RMSE: {rmse:.4f}")

    fold_num += 1

# 5) Durchschnittlicher RMSE
avg_rmse = np.mean(rmse_list)
print(f"\nDurchschnittlicher RMSE über {n_splits} Folds: {avg_rmse:.4f}")

# Optional: Plotten des letzten Folds (Test vs. Prediction)
# Wir können z.B. X_test / y_test / y_pred aus dem letzten Fold verwenden
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test, label="Tatsächliche Werte")
plt.plot(range(len(y_pred)), y_pred, label="XGB Vorhersagen", alpha=0.7)
plt.title("XGBoost - Letzter Fold: Echte vs. vorhergesagte Werte")
plt.xlabel("Test Samples")
plt.ylabel("Close Price")
plt.legend()
plt.show()
