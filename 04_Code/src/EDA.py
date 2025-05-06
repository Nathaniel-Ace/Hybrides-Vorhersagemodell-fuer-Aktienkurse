import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

# Hochgeladene Dateien
paths = {
    "NVDA": "../../03_Daten/processed_data/merged_weekly_NVDA_2015-2025.csv",
    "GOOG": "../../03_Daten/processed_data/merged_weekly_GOOG_2015-2025.csv",
    "MSFT": "../../03_Daten/processed_data/merged_weekly_MSFT_2015-2025.csv"
}

# GTD-Spalten
gtd_cols = {
    "NVDA": ["NVIDIA stock", "buy NVIDIA stock", "sell NVIDIA stock"],
    "GOOG": ["Google stock", "buy Google stock", "sell Google stock"],
    "MSFT": ["Microsoft stock", "buy Microsoft stock", "sell Microsoft stock"]
}

for ticker, path in paths.items():
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    df["Return"] = df["Close"].pct_change()

    print(f"\n=== {ticker}: Beschreibung ===")
    print(df.describe())

    print("Fehlende Werte:")
    print(df.isnull().sum())

    # Plot: Schlusskurs
    plt.plot(df["Close"])
    plt.title(f"{ticker} – Wöchentlicher Schlusskurs")
    plt.xlabel("Datum"); plt.ylabel("Close")
    plt.tight_layout(); plt.show()

    # Plot: Histogramm Close
    sns.histplot(df["Close"].dropna(), kde=True)
    plt.title(f"{ticker} – Verteilung Schlusskurs")
    plt.xlabel("Close")
    plt.tight_layout(); plt.show()

    # Plot: Boxplot Close
    sns.boxplot(x=df["Close"].dropna())
    plt.title(f"{ticker} – Boxplot Schlusskurs")
    plt.xlabel("Close")
    plt.tight_layout(); plt.show()

    # Plot: Wöchentliche Renditen
    plt.plot(df["Return"])
    plt.title(f"{ticker} – Wöchentliche Rendite")
    plt.xlabel("Datum"); plt.ylabel("Return")
    plt.tight_layout(); plt.show()

    # Korrelationsmatrix zwischen Close und GTD
    corr_cols = ["Close"] + gtd_cols[ticker]
    corr_df = df[corr_cols].dropna()
    corr_matrix = corr_df.corr()

    print("Korrelationsmatrix:")
    print(corr_matrix)

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{ticker} – Korrelation: Schlusskurs und Google Trends")
    plt.tight_layout(); plt.show()
