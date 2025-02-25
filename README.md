# Bachelorarbeit: Ein hybrider Ansatz zur Aktienkursprognose

Dieses Repository enthält alle relevanten Dateien und Informationen zur Bachelorarbeit mit dem Titel:

**„Ein hybrider Ansatz zur Aktienkursprognose: Kombination von Machine Learning und Google Trends für NVIDIA, Google und Microsoft“**

---

## Autor und Betreuung

- **Autor:** Nathaniel Ace Panganiban  
- **Matrikelnummer:** 2210257040  
- **Betreuer:** Christian Brandstätter  
- **Studiengang:** Informatik / Computer Science  
- **Datum:** Wien, 18.02.2025

---

## Überblick

Die Bachelorarbeit beschäftigt sich mit der Entwicklung eines Hybridmodells zur Vorhersage von Aktienkursen, bei dem traditionelle Finanzmodelle mit modernen Machine Learning-Methoden und alternativen Datenquellen, insbesondere Google Trends, kombiniert werden. Der Ansatz zielt darauf ab, die Vorhersagegenauigkeit zu erhöhen, indem historische Marktbewegungen und aktuelle Suchtrends simultan berücksichtigt werden.

---

## Motivation

Traditionelle Modelle zur Aktienkursprognose basieren meist ausschließlich auf historischen Preisdaten und technischen Indikatoren. Dies führt zu folgenden Herausforderungen:
- **Fehlende externe Einflussfaktoren:** Klassische Modelle integrieren keine zusätzlichen Daten, die das Investorenverhalten oder Markttrends widerspiegeln.
- **Begrenzte Mustererkennung:** Lineare Modelle können oft keine komplexen, nichtlinearen Zusammenhänge erfassen.
- **Mangelnde Interpretierbarkeit:** Viele Machine Learning-Modelle liefern präzise Vorhersagen, jedoch ohne transparente Einsicht in die zugrundeliegenden Entscheidungsprozesse.

Die Einbindung von Google Trends-Daten soll als Frühindikator für Kursbewegungen dienen und eine dynamische Anpassung der Prognosen ermöglichen. Zusätzlich wird Explainable AI (XAI) in Form von SHAP-Werten verwendet, um die Relevanz der einzelnen Merkmale transparent zu machen.

---

## Forschungsfragen

Die Arbeit zielt darauf ab, die folgenden zentralen Fragestellungen zu beantworten:

1. **Modellauswahl und Prognosegenauigkeit:**  
   Wie beeinflusst die Wahl des Machine Learning-Modells (LSTM vs. XGBoost) die Vorhersagegenauigkeit des Hybridmodells?

2. **Kombination von Modellen:**  
   Kann die Integration von GARCH mit LSTM/XGBoost die Prognosegenauigkeit für die Aktien von NVIDIA, Google und Microsoft verbessern?

3. **Relevanz alternativer Daten:**  
   Wie wichtig sind historische Finanzdaten im Vergleich zu alternativen Datenquellen für ein hybrides Prognosemodell?

4. **Korrelation von Google Trends und Aktienkursen:**  
   Wie stark korrelieren die Google Trends-Daten mit den Aktienkursen der genannten Unternehmen?

---

## Methodik

Die Umsetzung des Hybridmodells erfolgt in mehreren Schritten:

- **Datenbasis:**  
  Verwendung historischer Aktienkursdaten (2015–2024) von Finanzplattformen wie Yahoo Finance, ergänzt durch technische Indikatoren (z. B. gleitende Durchschnitte, RSI) und Google Trends-Daten (abgerufen mittels der pytrends-Bibliothek).

- **Modellimplementierung:**  
  - **GARCH-Modell:** Erfassung und Prognose von Volatilitäten zur Risikobewertung.  
  - **Machine Learning-Ansätze:** Einsatz von LSTM-Netzwerken bzw. XGBoost zur Identifikation nichtlinearer Muster und langfristiger Abhängigkeiten.  
  - **Ensemble-Techniken:** Kombination der Modelle, um die jeweiligen Stärken zu vereinen.

- **Evaluierungsmethoden:**  
  - Statistische Fehlermetriken (MSE, RMSE) zur Quantifizierung der Abweichungen zwischen prognostizierten und tatsächlichen Kursen.  
  - Finanzmetriken wie das Sharpe Ratio zur Bewertung der Prognosen im Kontext von Handelsstrategien.  
  - Einsatz von SHAP zur Erklärung und Visualisierung der Einflussfaktoren.

---

## Erwartete Ergebnisse

Die Arbeit rechnet mit folgenden Ergebnissen:

- **Verbesserte Prognosegenauigkeit:**  
  Durch die Kombination von GARCH mit modernen Machine Learning-Modellen wird eine signifikante Steigerung der Vorhersagegenauigkeit erwartet im Vergleich zu herkömmlichen statistischen Ansätzen (wie ARIMA).

- **Identifikation relevanter Einflussfaktoren:**  
  Es wird erwartet, dass bestimmte Google Trends-Suchbegriffe starke Korrelationen zu den Kursbewegungen aufweisen.

- **Transparenz durch XAI:**  
  Die Nutzung von SHAP soll die Interpretierbarkeit der Modellentscheidungen erhöhen und Einblicke in die wichtigsten Einflussfaktoren geben.

- **Potenzial für algorithmische Handelsstrategien:**  
  Das entwickelte Modell könnte als Grundlage für automatisierte Handelsstrategien dienen, indem es Marktbewegungen frühzeitig erkennt.

---

## Referenzen

Die Bachelorarbeit stützt sich auf eine fundierte Auswahl an Fachliteratur, u.a.:
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. John Wiley & Sons. :contentReference[oaicite:0]{index=0}
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. :contentReference[oaicite:1]{index=1}
- Preis, T., Moat, H. S., & Stanley, H. E. (2013). *Quantifying trading behavior in financial markets using Google Trends*. :contentReference[oaicite:2]{index=2}
- Weitere relevante Quellen finden sich im Literaturverzeichnis der Arbeit.

---

## Nutzung und Weiteres Vorgehen

- **Projektdokumentation:**  
  Dieses README dient als Einstieg und Überblick über die Bachelorarbeit. Alle relevanten Dokumente, Code-Snippets (sofern vorhanden) und weitere Materialien sind in diesem Repository strukturiert abgelegt.

- **Qualitätssicherung:**  
  Der Prozess zur Erstellung und Evaluation der Arbeit erfolgt nach einem strukturierten Prozess mit definierten Quality Gates (Proposal und Eigenanteil) gemäß den Vorgaben der Fakultät Computer Science & Applied Mathematics.
 

