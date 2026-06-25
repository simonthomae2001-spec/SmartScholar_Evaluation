# SmartScholar — Testprotokoll (Termin 3)

**Projekt:** SmartScholar — Agentic RAG Research Assistent  
**Dokumentenklasse:** Praktikums-Deliverable (Termin 3)  
**Status:** Erste End-to-End-Durchläufe erfolgreich durchgeführt  

---

## 1. Testdurchläufe nach Forschungsprofilen

Nach der Zusammenführung der einzelnen Agenten wurde das Gesamtsystem in den drei vordefinierten Modi (**Fast**, **Medium**, **Pro**) mit einer Standard-Testanfrage getestet. Die Tests wurden auf einem leistungsfähigen lokalen Entwicklungsrechner (Ollama) durchgeführt.

* **Fast-Modus:** Läuft erfolgreich durch. Die Rechendauer ist moderat bis lang, da ausschließlich Abstracts analysiert werden.
* **Medium-Modus:** Läuft erfolgreich durch. Die Rechendauer erhöht sich durch die größere Anzahl an Suchanfragen und Papern.
* **Pro-Modus:** Läuft erfolgreich durch. Die Rechendauer ist sehr lang, da die Verarbeitung der vollständigen Texte extrem rechenintensiv für die lokale Hardware ist.

**Fazit der Modi:** Das System funktioniert in allen drei Profilen grundlegend stabil. Die Rechendauer variiert stark je nach Modus und ist selbst auf potenter Hardware im oberen Bereich anzusiedeln.

---

## 2. Aufgetretene Fehler & Auffälligkeiten

Bei den ersten vollständigen Durchläufen sind zwei konkrete Punkte aufgefallen, die für die nächsten Versionen optimiert werden müssen:

### 2.1 JSON-Formatierungsproblem (Gelegentliche Exceptions)
Es kommt unregelmäßig zu Programmabbrüchen innerhalb der Agenten-Pipeline (z. B. beim Researcher- oder Analyst-Agenten). 
* **Ursache:** Das lokale LLM hält sich gelegentlich nicht exakt an die geforderte JSON-Struktur. 
* **Auswirkung:** Das System kann die Antwort nicht parsen und wirft eine Exception, was zum Abbruch des Durchlaufs führt.

### 2.2 Sprachverhalten (Qualität bei Nicht-Englischen Anfragen)
Das System ist grundsätzlich in der Lage, mit anderen Sprachen (getestet mit Deutsch) umzugehen.
* **Auffälligkeit:** Bei deutschen Anfragen schleichen sich im finalen Bericht (Synthesizer) häufiger Grammatik- und Satzbaufehler ein. Das System performt im Englischen aktuell deutlich fehlerfreier und flüssiger, da die zugrundeliegenden Prompts und das lokale Modell (llama3) scheinbar stark auf Englisch optimiert sind.