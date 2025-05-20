import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Einstellungen
ticker       = "NVDA"
window_size  = 10
n_splits     = 3
csv_path     = f"../../03_Daten/processed_data/merged_weekly_{ticker}_2015-2025_with_trends.csv"

# 1) Daten einlesen und Log-Renditen berechnen
df = (
    pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
      .sort_index()
)
df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# 2) Feature-Erzeuger
def create_features_and_target(df, window_size=10):
    X, y = [], []
    rets = df["Return"].values
    for i in range(window_size, len(df)):
        # a) Sequenz der letzten Log-Renditen
        seq = list(rets[i-window_size:i])
        # b) statische Features zum Zeitpunkt i
        garch_vol = df["GARCH_vol"].iat[i]
        rsi       = df["RSI_14"].iat[i]
        trend_avg = df["Trend_Average"].iat[i]
        trend_sm  = df["Trend_Smoothed"].iat[i]
        X.append(seq + [garch_vol, rsi, trend_avg, trend_sm])
        y.append(rets[i])
    return np.array(X), np.array(y)

# 3) Vollständiges Feature-Set für Hyperparameter-Tuning
#    (GARCH-Volatilität muss zuerst einmalig über gesamten Datensatz berechnet werden)
scaled_all = df["Return"] * 10
gmod_all   = arch_model(scaled_all, mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off")
df["GARCH_vol"] = gmod_all.conditional_volatility / 10

X_full, y_full = create_features_and_target(df, window_size)
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# 4) RandomizedSearchCV mit TimeSeriesSplit
param_dist = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
    "gamma":           [0, 1, 5]
}
tscv_tune = TimeSeriesSplit(n_splits=3)
base_model = XGBRegressor(
    tree_method="hist", predictor="cpu_predictor",
    verbosity=0, random_state=42
)
rnd_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv_tune,
    scoring="neg_mean_squared_error",
    random_state=42,
    verbose=1,
    error_score="raise"
)
rnd_search.fit(X_full_scaled, y_full)
best_params = rnd_search.best_params_

print(">>> Best XGBoost-Params:", best_params)

# 5) Finaler CV-Durchlauf mit den gefundenen Parametern + Early Stopping
tscv = TimeSeriesSplit(n_splits=n_splits)
rmse_returns, rmse_prices = [], []
last_fold = {}

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # a) GARCH fitten und Forecast
    scaled = train_df["Return"] * 10
    gmod   = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off")
    train_df["GARCH_vol"] = gmod.conditional_volatility / 10
    fc = gmod.forecast(start=train_df.index[-1], horizon=len(test_df), reindex=False)
    test_df["GARCH_vol"] = np.sqrt(fc.variance.values[-1, :]) / 10

    # b) Features und Targets erzeugen
    X_train, y_train = create_features_and_target(train_df, window_size)
    X_test,  y_test  = create_features_and_target(test_df,  window_size)

    # c) Skalieren
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Fold {fold}: X_train={X_train.shape}, X_test={X_test.shape}")

    # d) XGBoost mit Early Stopping
    model = XGBRegressor(
        **best_params,
        tree_method="hist", predictor="cpu_predictor",
        verbosity=0, random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    y_pred = model.predict(X_test)

    # e) RMSE Log-Renditen
    rm_r = math.sqrt(mean_squared_error(y_test, y_pred))
    rmse_returns.append(rm_r)

    # f) Rückrechnung in Close-Preise & RMSE
    preds, actuals = [], []
    for i, r in enumerate(y_pred):
        idx    = i + window_size
        p_prev = test_df["Close"].iloc[idx-1]
        p_pred = p_prev * np.exp(r)
        preds.append(p_pred)
        actuals.append(test_df["Close"].iloc[idx])
    rm_p = math.sqrt(mean_squared_error(actuals, preds))
    rmse_prices.append(rm_p)

    print(f"Fold {fold}: RMSE_Returns={rm_r:.4f}, RMSE_Prices={rm_p:.4f}")

    if fold == n_splits:
        last_fold = {
            "idx":     test_df.index[window_size:],
            "y_test":  y_test,
            "y_pred":  y_pred,
            "actuals": actuals,
            "preds":   preds
        }

# 6) Endgültige Ergebnisse
print(f"\nDurchschn. RMSE Log-Renditen: {np.mean(rmse_returns):.4f}")
print(f"Durchschn. RMSE Aktienkurse: {np.mean(rmse_prices):.4f}")

# 7) Plots für den letzten Fold
idx = last_fold["idx"]
plt.figure(figsize=(12,5))
plt.plot(idx, last_fold["y_test"], label="Actual Log-Renditen")
plt.plot(idx, last_fold["y_pred"], "--", label="Pred Log-Renditen", alpha=0.7)
plt.title(f"{ticker} – Fold {n_splits}: Log-Renditen")
plt.xlabel("Datum"); plt.ylabel("Log-Rendite"); plt.legend(); plt.show()

plt.figure(figsize=(12,5))
plt.plot(idx, last_fold["actuals"], label="Actual Close")
plt.plot(idx, last_fold["preds"],   "--", label="Pred Close", alpha=0.7)
plt.title(f"{ticker} – Fold {n_splits}: Aktienkurse")
plt.xlabel("Datum"); plt.ylabel("Preis (Close)"); plt.legend(); plt.show()
