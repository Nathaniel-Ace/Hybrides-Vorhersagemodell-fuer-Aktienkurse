import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

# Pfade
base_path   = "../../03_Daten/processed_data/merged_weekly_NVDA_2015-2025.csv"
trend_path  = "../../03_Daten/processed_data/merged_weekly_NVDA_2015-2025_with_trends.csv"

# 1) Einlesen
base_df  = pd.read_csv(base_path,  parse_dates=["Date"], index_col="Date").sort_index()
trend_df = pd.read_csv(trend_path, parse_dates=["Date"], index_col="Date").sort_index()

# 2) Log-Returns berechnen (für GARCH)
base_df["Return"] = np.log(base_df["Close"] / base_df["Close"].shift(1))
base_df.dropna(subset=["Return"], inplace=True)

# 3) GARCH auf der kompletten Return-Serie fitten und Volatilität ableiten
#    (skaliere returns *10 und rescale=False analog Training)
am = arch_model(base_df["Return"] * 10, mean="Zero", vol="GARCH", p=1, q=1,
                dist="normal", rescale=False)
res = am.fit(disp="off")
# konditionale Volatilität zurück auf Originalskala
base_df["GARCH_vol"] = res.conditional_volatility / 10

# 4) Zusammenführen aller Trend-Spalten
cols_trend = ["Trend_Average", "Trend_Smoothed"]
df = base_df.join(trend_df[cols_trend], how="inner")

# 5) GTD-Spalten detektieren
gtd_cols = [c for c in base_df.columns if "stock" in c.lower()]

# 6) Feature-Liste
features = ["GARCH_vol", "RSI_14"] + gtd_cols + cols_trend

# 7) Korrelation berechnen
corr = df[features].corr()

# 8) Ausgabe
print("Korrelationsmatrix (gerundet):")
print(corr.round(2))

# 9) Heatmap plotten
plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    cbar_kws={"shrink":.8},
    square=True
)
plt.title("Feature-Korrelationen NVDA 2015–2025")
plt.tight_layout()
plt.show()
