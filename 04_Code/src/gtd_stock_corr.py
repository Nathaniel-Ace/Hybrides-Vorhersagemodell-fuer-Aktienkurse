import pandas as pd
import matplotlib.pyplot as plt

# Ticker-konfiguration
ticker_files = {
    "NVDA": {
        "file": "../../03_Daten/processed_data/merged_weekly_NVDA_2015-2025.csv",
        "gtd": ["NVIDIA stock", "buy NVIDIA stock", "sell NVIDIA stock"]
    },
    "GOOG": {
        "file": "../../03_Daten/processed_data/merged_weekly_GOOG_2015-2025.csv",
        "gtd": ["Google stock", "buy Google stock", "sell Google stock"]
    },
    "MSFT": {
        "file": "../../03_Daten/processed_data/merged_weekly_MSFT_2015-2025.csv",
        "gtd": ["Microsoft stock", "buy Microsoft stock", "sell Microsoft stock"]
    },
}

for ticker, cfg in ticker_files.items():
    df = pd.read_csv(cfg["file"], parse_dates=["Date"], index_col="Date").sort_index()

    # 4-Wochen-Rolling-Mean der GTD
    df_roll = df[cfg["gtd"]].rolling(window=4, min_periods=1).mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df["Close"], color="tab:blue", linewidth=2, label="Close Price")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Close Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    styles = ["--", "-.", ":"]
    for col, ls in zip(cfg["gtd"], styles):
        ax2.plot(df_roll.index, df_roll[col],
                 linestyle=ls, alpha=0.8, label=f"{col} (4-Wochen MA)")
    ax2.set_ylabel("Google Trends (4-Wochen gleitend)")

    # Legenden an getrennten Ecken
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper left")
    ax2.legend(h2, l2, loc="upper right")

    plt.title(f"{ticker} – Kurs vs. Google Trends (4-Wochen-MA)")
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()
