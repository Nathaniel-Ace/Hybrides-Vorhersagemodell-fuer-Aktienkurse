import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# ————— Einstellungen —————
ticker      = "NVDA"
window_size = 10
n_splits    = 3
csv_path    = f"../../03_Daten/processed_data/merged_weekly_{ticker}.csv"

# 1) Daten einlesen und Log-Renditen berechnen
df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) GARCH auf gesamten Datensatz fitten (einmalig) und Volatilität speichern
g = arch_model(df["Return"] * 10, mean="Zero", vol="GARCH", p=1, q=1,
               dist="normal", rescale=False)
res_all = g.fit(disp="off")
df["GARCH_vol"] = res_all.conditional_volatility / 10

# 3) Statische Feature-Spalten ermitteln
gtd_cols     = [c for c in df.columns if "stock" in c.lower()]
static_cols  = ["GARCH_vol", "RSI_14"] + gtd_cols

# 4) Funktion zum Erzeugen von X,y
def make_xy(subdf):
    X, y = [], []
    rets = subdf["Return"].values
    for i in range(window_size, len(subdf)):
        seq   = list(rets[i-window_size:i])                   # letzte Log-Renditen
        stat  = subdf[static_cols].iloc[i].values.tolist()     # GARCH_vol, RSI, GTD
        X.append(seq + stat)
        y.append(rets[i])
    return np.array(X), np.array(y)

# 5) Hyperparameter-Suche für XGBoost
X_full, y_full = make_xy(df)

tscv = TimeSeriesSplit(n_splits=n_splits)
param_dist = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "gamma":           [0, 0.1, 0.5]
}
xgb = XGBRegressor(random_state=42, tree_method="hist")
search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=20, cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_full, y_full)
best_params = search.best_params_
print(">>> Best XGBoost Params:", best_params)

# 6) Out-of-Sample-Evaluation mit Early-Stopping
rmse_ret   = []
rmse_price = []
last_fold  = {}

for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # a) GARCH-Forecast für Test-Set
    fc = res_all.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # b) Skalierung der statischen Features
    scaler = StandardScaler()
    train_df[static_cols] = scaler.fit_transform(train_df[static_cols])
    test_df[static_cols]  = scaler.transform(test_df[static_cols])

    # c) Features/Targets
    X_train, y_train = make_xy(train_df)
    X_test,  y_test  = make_xy(test_df)
    print(f"Fold {fold}: X_train={X_train.shape}, X_test={X_test.shape}")

    # d) Finales Training mit Early Stopping auf Test-Set
    model = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=10,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # e) Vorhersage & RMSE auf Log-Renditen
    y_pred = model.predict(X_test)
    r_ret  = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_ret.append(r_ret)

    # f) Rückrechnung auf Kursbasis & RMSE Preise
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        p0 = test_df["Close"].iat[i + window_size - 1]
        preds.append(p0 * np.exp(r))
        actuals.append(test_df["Close"].iat[i + window_size])
    r_price = math.sqrt(mean_squared_error(actuals, preds))
    rmse_price.append(r_price)

    print(f"Fold {fold}: RMSE_Returns={r_ret:.4f}, RMSE_Prices={r_price:.4f}")

    if fold == n_splits:
        last_fold = {
            "idx":    test_df.index[window_size:],
            "y_test": y_test,
            "y_pred": y_pred,
            "actual": actuals,
            "preds":  preds
        }

# 7) Durchschnittsergebnisse
print("\n--- Durchschnitt über alle Folds ---")
print("Log-Renditen RMSE:", np.mean(rmse_ret))
print("Aktienkurse RMSE:", np.mean(rmse_price))

# 8) Plots für den letzten Fold
if last_fold:
    idx = last_fold["idx"]
    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["y_test"], label="Real Log-Renditen")
    plt.plot(idx, last_fold["y_pred"], "--", label="Pred Log-Renditen", alpha=0.7)
    plt.title(f"{ticker} – Fold {n_splits} Log-Renditen")
    plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend(); plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(idx, last_fold["actual"], label="Real Close")
    plt.plot(idx, last_fold["preds"],   "--", label="Pred Close", alpha=0.7)
    plt.title(f"{ticker} – Fold {n_splits} Aktienkurse")
    plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.show()
