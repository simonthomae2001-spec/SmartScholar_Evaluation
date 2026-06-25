# SmartScholar — Testprotokoll & Edge-Case-Behebung (Termin 4)

**Projekt:** SmartScholar — Agentic RAG Research Assistent  
**Dokumentenklasse:** Praktikums-Deliverable (Termin 4)  
**Fokus:** Absicherung der Pipeline, Fehlerbehandlung und UI/UX-Optimierung  
**Status:** Abgeschlossen und stabilisiert  

---

## 1. Übersicht

Nachdem in Termin 3 die grundlegenden End-to-End-Durchläufe der Profile evaluiert wurden, lag der Fokus für **Termin 4** auf der Härtung des Systems. Es wurden gezielt kritische Edge-Cases (Fehlbedienungen, API-Ausfälle, Sprachbarrieren) provoziert, dokumentiert und durch entsprechende Code-Anpassungen softwareseitig abgefangen.

---

## 2. Edge-Case-Protokollierung & Lösungsansätze

### Edge-Case 2.1: Keine Paper in der UI ausgewählt
Wenn das System dem Nutzer im Human-in-the-Loop-Schritt (HITL) die gefundenen Paper zur Auswahl vorlegt, besteht das Risiko, dass der Nutzer alle Haken entfernt und auf "Weiter" klickt.

* **Erwartetes Verhalten (Soll):** Das System bricht den Durchlauf nicht ab, sondern zeigt dem Nutzer eine klare Fehlermeldung in der UI, dass mindestens ein Element ausgewählt werden muss, um fortzufahren.
* **Tatsächliches Verhalten (Ist):** Die Pipeline lief mit einer leeren Liste in den nächsten Agenten (Analyst/Synthesizer). Da keine Daten vorhanden waren, stürzte das System im Backend mit einem Error ab.
* **Gewählter Lösungsansatz:** In `app.py` wurde vor der Übergabe an den LangGraph-State eine Validierung eingebaut. Wenn `len(selected_papers) == 0` ist, wird die Weiterleitung blockiert und über `st.error()` eine Fehlermeldung ausgegeben. Das System stürzt nicht ab und kann für weitere Anfragen genutzt werden.

---

### Edge-Case 2.2: Nicht sinnvolle / Off-Topic Anfragen
Eingabe von Prompts, die keinen wissenschaftlichen Bezug haben (z. B. "Wie repariere ich ein Fahrrad?").

* **Erwartetes Verhalten (Soll):** Der `GatekeeperAgent` erkennt, dass es sich um keine akademische Forschungsfrage handelt, blockiert den Start der Suche und gibt Feedback.
* **Tatsächliches Verhalten (Ist):** Der Gatekeeper war bisher als reiner struktureller Dummy (Stub) implementiert und hat jede Anfrage bedingungslos durchgewinkt. Das System hat daraufhin versucht, auf Semantic Scholar nach unpassenden Begriffen zu suchen, was zu leeren oder unbrauchbaren Ergebnissen führte.
* **Gewählter Lösungsansatz:** Der `GatekeeperAgent` wurde ausprogrammiert. Er nutzt nun ein schlankes System-Prompt im lokalen LLM, welches die Anfrage klassifiziert (Wissenschaftlich: Ja/Nein). Erkennt das Modell ein "Nein", bricht der Graph sofort im ersten Schritt ab und die UI spiegelt dies mit einer entsprechenden Meldung wider. Das System bleibt stabil und kann weiter genutzt werden.

---

### Edge-Case 2.3: Kein Paper wird gefunden / API-Anfrage fehlgeschlagen
Die Semantic Scholar API liefert entweder aufgrund von Netzwerkproblemen (z. B. HTTP 500 / Rate Limit erreicht) oder weil es schlicht keine Literatur zum Thema gibt, 0 Ergebnisse zurück.

* **Erwartetes Verhalten (Soll):** Eine saubere, für den Nutzer spezifizierte Fehlermeldung, warum die Suche nicht fortgesetzt werden kann, ohne dass die Streamlit-Oberfläche einfriert.
* **Tatsächliches Verhalten (Ist):** Das `ScholarTool` warf bei einem API-Fehler eine ungefangene `HTTPError`-Exception. Gab es einfach nur 0 Treffer, crashte der nachfolgende `IngestorAgent` beim Versuch, Daten zu verarbeiten, die gar nicht existierten.
* **Gewählter Lösungsansatz:** Das Programm wurde für diesen Case abgesichert. Zudem prüft der `ResearcherAgent` nun explizit das Resultat. Sind keine Paper vorhanden, wird der Graph kontrolliert gestoppt und die UI gibt eine generische Meldung aus. Diese könnte in der Zukunft noch verfeinert werden.

---

### Edge-Case 2.4: UI-Buttons drücken während das System rechnet
Da die lokalen LLM-Aufrufe viel Zeit in Anspruch nehmen, neigen Nutzer dazu, ungeduldig mehrfach auf Buttons (z. B. "Suche starten") zu klicken.

* **Erwartetes Verhalten (Soll):** Während einer laufenden Berechnung müssen alle Interaktionselemente gesperrt sein, um den Zustand des Graphs nicht zu korrumpieren.
* **Tatsächliches Verhalten (Ist):** Die Buttons wurden von Anfang an auf inaktiv während der Berechnung gesetzt. So können Nutzer ausschließlich den Prozess abbrechen und an den vorgesehen Stellen Bulletpoints oder Paper für die Literature Review auswählen. Sonstiges Interagieren, welches zu unerwünschtem Verhalten führen könnte ist nicht möglich.
* **Gewählter Lösungsansatz:** /

---

### Edge-Case 2.5: Grammatikalische Defizite bei nicht-englischen Anfragen (Deutsch)
Das System wird mit einer deutschen Forschungsfrage gefüttert.

* **Erwartetes Verhalten (Soll):** Das System verarbeitet die Anfrage und generiert den finalen Forschungsbericht in grammatikalisch einwandfreiem Deutsch.
* **Tatsächliches Verhalten (Ist):** Der finale Bericht des `SynthesizerAgent` wies häufig Satzbaufehler, falsche Artikel oder einen unnatürlichen Mix aus englischen Fachbegriffen und deutscher Grammatik auf ("Denglisch"). Da die Prompts des Synthesizers primär auf Englisch verfasst waren, "verrutschte" das lokale Sprachmodell (llama3) sprachlich im Output.
* **Gewählter Lösungsansatz:** Noch konnten wir diesen Case nicht abfangen. In späteren Versionen muss getestet werden, ob ein fehlerhafter Text von der LLM erneut überarbeitet werden kann.

---

## 3. Fazit & Projektabschluss

Durch die Behebung dieser vier zentralen Edge-Cases hat das System in Termin 4 ein hohes Maß an Produktionsreife erlangt. Die Anwendung fängt Fehlbedienungen im UI elegant ab und stürzt bei API-Engpässen nicht mehr ab. Somit liefert das System bereits stabil Ergebnisse.