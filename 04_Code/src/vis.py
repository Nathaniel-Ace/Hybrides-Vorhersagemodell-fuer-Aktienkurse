import matplotlib.pyplot as plt

# Modellvarianten
models = ["LSTM", "XGBoost", "G+L", "G+L+RSI", "G+L+RSI+GTD", "G+X", "G+X+RSI", "G+X+RSI+GTD"]
x = range(len(models))

# RMSE-Werte Preisprognose (Schlusskurse)
rmse_nvda = [6.6339, 27.0426, 2.7375, 2.3375, 2.8136, 2.8578, 2.4853, 2.4625]
rmse_goog = [9.4500, 30.0505, 3.9578, 3.5819, 6.2158, 4.4258, 3.7953, 3.6881]
rmse_msft = [18.9792, 80.4302, 7.6404, 6.9672, 14.3669, 8.0835, 7.2955, 7.1738]

# === Visualisierung 1: Reine ML-Modelle vs. GARCH-Hybride ===
plt.figure(figsize=(12, 6))
plt.scatter(x, rmse_nvda, color='green', label="NVDA", marker='o')
plt.scatter(x, rmse_goog, color='blue', label="GOOG", marker='s')
plt.scatter(x, rmse_msft, color='orange', label="MSFT", marker='^')
plt.title("RMSE – Reine ML-Modelle vs. GARCH-Hybride (Preisprognose)")
plt.xlabel("Modellvariante")
plt.ylabel("RMSE")
plt.xticks(x, models, rotation=30)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Modellvarianten der GARCH-Hybride
models_hybrid = ["G+L", "G+L+RSI", "G+L+RSI+GTD", "G+X", "G+X+RSI", "G+X+RSI+GTD"]
x_hybrid = range(len(models_hybrid))

# RMSE-Werte Preisprognose aus den Tabellen
rmse_nvda = [2.7375, 2.3375, 2.8136, 2.8578, 2.4853, 2.4625]
rmse_goog = [3.9578, 3.5819, 6.2158, 4.4258, 3.7953, 3.6881]
rmse_msft = [7.6404, 6.9672, 14.3669, 8.0835, 7.2955, 7.1738]

# Diagramm mit Punkten (kein Liniendiagramm)
plt.figure(figsize=(10, 6))
plt.scatter(x_hybrid, rmse_nvda, color='green', label="NVDA", marker='o')
plt.scatter(x_hybrid, rmse_goog, color='blue', label="GOOG", marker='s')
plt.scatter(x_hybrid, rmse_msft, color='orange', label="MSFT", marker='^')

# Achsenbeschriftungen und Formatierung
plt.title("RMSE – Preisprognose: Einfluss von Features auf GARCH-Hybridmodelle")
plt.xlabel("Modellvariante")
plt.ylabel("RMSE")
plt.xticks(x_hybrid, models_hybrid, rotation=30)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
