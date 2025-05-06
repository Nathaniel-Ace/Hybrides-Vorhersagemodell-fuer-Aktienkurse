import matplotlib.pyplot as plt

# Modellvarianten
models = ["LSTM", "XGBoost", "G+L", "G+L+RSI", "G+L+RSI+GTD", "G+X", "G+X+RSI", "G+X+RSI+GTD"]

# RMSE-Werte aus deiner Tabelle (Preisebene)
rmse_nvda = [6.6339, 27.0426, 2.7375, 2.3375, 2.8136, 2.8578, 2.4853, 2.4625]
rmse_goog = [9.4500, 30.0505, 3.9578, 3.5819, 6.2158, 4.4258, 3.7953, 3.6881]
rmse_msft = [18.9792, 80.4302, 7.6404, 6.9672, 14.3669, 8.0835, 7.2955, 7.1738]

# Plot erstellen
plt.figure(figsize=(12, 6))
plt.plot(models, rmse_nvda, marker='o', color='green', label="NVDA")
plt.plot(models, rmse_goog, marker='o', color='blue', label="GOOG")
plt.plot(models, rmse_msft, marker='o', color='orange', label="MSFT")

# Achsentitel und Formatierung
plt.title("Vergleich von RMSE: Reine ML-Modelle vs. GARCH-Hybride (Preisprognose)")
plt.xlabel("Modellvariante")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
