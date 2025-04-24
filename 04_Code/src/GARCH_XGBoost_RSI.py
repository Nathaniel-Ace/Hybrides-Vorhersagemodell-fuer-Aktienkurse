import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# ————— Einstellungen —————
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/historical_stock_data_weekly_{ticker}_flat_with_RSI.csv"

# 1) Daten einlesen und Log-Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Einmalig GARCH auf gesamten Datensatz fitten und Volatilität speichern
res_all = arch_model(df["Return"] * 10, mean="Zero", vol="GARCH", p=1, q=1,
                     dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = res_all.conditional_volatility / 10

# 3) Features/Target erzeugen
def make_xy(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    vols = df["GARCH_vol"].values
    rsis = df["RSI_14"].values
    for i in range(window_size, len(df)):
        seq   = list(rets[i-window_size:i])
        feats = seq + [vols[i], rsis[i]]
        X.append(feats)
        y.append(rets[i])
    return np.array(X), np.array(y)

X_full, y_full = make_xy(df, window_size)

# 4) Hyperparameter-Suche
tscv = TimeSeriesSplit(n_splits=n_splits)
param_dist = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "gamma":           [0, 0.1, 0.5]
}

xgb = XGBRegressor(
    random_state=42,
    tree_method="hist"        # CPU-basierter Baum-Builder
)

search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=20,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_full, y_full)
best_params = search.best_params_
print(">>> Best XGBoost Params:", best_params)

# 5) Out-of-Sample Evaluation
rmse_ret   = []
rmse_price = []
last_fold  = None

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # GARCH-Forecast für Test-Set
    fc = res_all.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # Fold-Features
    X_train, y_train = make_xy(train_df, window_size)
    X_test,  y_test  = make_xy(test_df,  window_size)
    print(f"Fold {fold}: X_train={X_train.shape}, X_test={X_test.shape}")

    # Train finaler XGB mit den besten Parametern
    model = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train, verbose=False)

    # Vorhersage & RMSE Log-Renditen
    y_pred = model.predict(X_test)
    rm = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_ret.append(rm)

    # Rückrechnung in Preisbasis & RMSE Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iat[i + window_size - 1]
        preds.append(p0 * np.exp(r))
        actuals.append(test_df["Close"].iat[i + window_size])
    rp = math.sqrt(mean_squared_error(actuals, preds))
    rmse_price.append(rp)

    print(f"Fold {fold}: RMSE_Returns={rm:.4f}, RMSE_Prices={rp:.4f}")

    if fold == n_splits:
        last_fold = {
            "idx":   test_df.index[window_size:],
            "y_t":   y_test,
            "y_p":   y_pred,
            "act":   actuals,
            "preds": preds
        }

# 6) Durchschnittswerte
print("\n--- Durchschnitt über alle Folds ---")
print("Log-Renditen RMSE:",   np.mean(rmse_ret))
print("Aktienkurse RMSE:",     np.mean(rmse_price))

# 7) Plots für den letzten Fold
if last_fold:
    idx = last_fold["idx"]

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_t"],  label="Real Log-Renditen")
    plt.plot(idx, last_fold["y_p"],  label="Pred Log-Renditen", alpha=0.7)
    plt.title("Letzter Fold – Log-Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend(); plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["act"],   label="Real Kurs")
    plt.plot(idx, last_fold["preds"], label="Pred Kurs", alpha=0.7)
    plt.title("Letzter Fold – Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.show()
