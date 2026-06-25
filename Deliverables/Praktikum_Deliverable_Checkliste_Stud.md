# Deliverable-Checkliste – Praktikum „Agentic AI, Multi-Agenten-Systeme und LLMs"

**Modul:** Agentic AI – Master · HAW Hamburg
**Zweck:** Vollständigkeitskontrolle aller Pflicht-Deliverables über die sechs Termine plus Abschlusspräsentation.
**Verwendung:** Jede Gruppe hakt am Ende eines Termins die erledigten Punkte ab. Der Dozent nutzt dieselbe Liste zur Abnahme.

---

## Vorab auszufüllen

- **Gruppe / Teamname:** SmartScholar
- **Gruppenmitglieder:** Oliver Tano Schlichting, Simon Thomae, Tobias Haugg, Shoaib Amiri, Ahmed Radwan
- **Gewähltes Projekt (A–F bzw. eigener Vorschlag):** Projekt C
- **Repository-Link:** https://github.com/TanOlive/SmartScholar

> Hinweis: Eigene Projektvorschläge müssen an mindestens drei Vorlesungskapitel anschließen und vorab mit dem Dozenten abgestimmt sein.

---

## Termin 1 – Kickoff und Setup

**Ziel:** Gruppe steht, Projekt gewählt, Repository existiert, Tooling bei allen lokal lauffähig.

- [x] Gruppe gebildet (bis zu 5 Personen)
- [x] Projekt aus dem Katalog gewählt **und beim Dozenten gemeldet**
- [x] Gemeinsames Git-Repository angelegt; alle Mitglieder haben Schreibzugriff
- [x] Rollen in der Gruppe verteilt (Repo/CI, Doku, Dozentenkontakt)
- [x] Tooling-Frage geklärt (LLM-Zugriff + Frameworks) und Basis-Pakete installiert

**Pflicht-Deliverable Termin 1**

- [x] `README.md` im Repo enthält: Projektname, Gruppenmitglieder, gewähltes Projekt, einen Absatz „Was wollen wir bauen?"
- [x] Erste **Architekturskizze** im README (als Foto oder Diagramm): Agenten/Komponenten, Kommunikation, benötigte externe Tools/APIs
- [x] „Hello, LLM"-Skript läuft **bei allen Gruppenmitgliedern lokal**

---

## Termin 2 – Architektur und Prototyp

**Ziel:** Aus der Skizze wird ein bewusst entworfenes Design; ein minimaler End-to-End-Prototyp läuft.

- [x] Architektur verfeinert, folgende Fragen **schriftlich** beantwortet:
  - [x] Aus welchen Komponenten besteht das System?
  - [x] Welche davon sind „Agenten" (Kap. 01)? Zuordnung zur Russell/Norvig-Taxonomie
  - [x] Wo liegen Memory, Planning, Action im Vier-Schichten-Modell?
  - [x] Wer trifft welche Entscheidungen – und warum dort?
- [x] Minimaler End-to-End-Durchlauf implementiert (Funktion vor Eleganz)
- [x] Erste Stolpersteine festgehalten (unklare Doku, Improvisationen, getroffene Annahmen)

**Pflicht-Deliverable Termin 2**

- [x] Architekturdokument (1–2 Seiten) im Repository
- [x] Lauffähiger Prototyp im Repository **mit Anleitung zum Starten**
- [x] Kurze Liste offener Fragen für die nächste Sitzung

---

## Termin 3 – Kernfunktionalität

**Ziel:** Das System tut, was es tun soll – zumindest im Standardfall.

- [x] Hauptlogik implementiert – projektspezifisch:
  - [ ] **Projekt A:** Vollständiger Agenten-Workflow vom Feature-Request bis zum getesteten Code
  - [ ] **Projekt B:** Spielmechanik + mind. zwei verschiedene Agenten-Typen, die gegeneinander spielen können
  - [x] **Projekt C:** Vollständige RAG-Pipeline: Suche → Retrieval → Synthese → strukturierter Output
  - [ ] **Projekt D:** Funktionierender Agent inkl. Tool-Zugriff, erste naive Guardrails, erste Angriffsversuche
  - [ ] **Projekt E:** Mind. drei Agenten mit Persönlichkeiten und persistentem Memory, die in mind. einem Szenario interagieren
  - [ ] **Projekt F:** Vollständiger Workflow für mind. einen Eingabe-Typ, mit klarem Input und Output
- [ ] Mehrere End-to-End-Durchläufe durchgeführt und protokolliert (Varianz beobachtet)
- [ ] Auffälliges/unerwartetes Systemverhalten notiert

**Pflicht-Deliverable Termin 3**

- [x] Lauffähige Kernfunktionalität im Repository
- [x] Kurzes „Lab Notebook" (Markdown-Datei im Repo) mit den Beobachtungen aus den Durchläufen

---

## Termin 4 – Erweiterung und Robustheit

**Ziel:** Das System überlebt ungewöhnliche Eingaben; das Innenleben ist nachvollziehbar.

- [x] **Observability** eingebaut: strukturiertes Logging aller LLM-Aufrufe (Prompt, Response, Latenz, Kosten)
- [x] Mind. **drei Edge Cases / Failure Modes** identifiziert (z. B. leere/absurde Eingabe, Tool-/API-Ausfall, Endlosschleife, falsches Output-Format)
- [x] Sinnvolle Fehlerbehandlung implementiert (bewusste Entscheidungen: wiederholen / abbrechen / an Nutzer melden)
- [ ] **Nur sicherheitsrelevante Projekte (D, A, F):** erste Guardrail-Schicht (Input-Filter, Output-Filter, Tool-Whitelist, Confirmation Gate für irreversible Aktionen)

**Pflicht-Deliverable Termin 4**

- [x] Observability-Setup ist aktiv und produziert Logs
- [x] Dokumentierte Edge-Case-Liste mit je: erwartetes Verhalten / tatsächliches Verhalten / gewählter Lösungsansatz
- [ ] Erste Version der Guardrails (sofern projekt-relevant)

---

## Termin 5 – Evaluation und Experiment

**Ziel:** Belegen, *wie gut* das System ist – und unter welchen Bedingungen es versagt.

- [ ] Mind. **eine quantitative und eine qualitative Metrik** definiert
- [ ] **Eine konkrete Hypothese** formuliert, die experimentell überprüft wird
- [ ] Experiment durchgeführt mit **mind. drei Durchläufen pro Variante**; alle Ergebnisse dokumentiert (auch unerwartete)
- [ ] Iteration durchgeführt: bei aufgedeckten Schwächen eine Komponente verbessert und erneut gemessen

**Pflicht-Deliverable Termin 5**

- [ ] Definition der Metriken im Repo
- [ ] Hypothesen-Dokument mit Versuchsaufbau und Ergebnissen (Tabellen, ggf. Plots)
- [ ] Schriftliche Reflexion: Was wurde gelernt – über das System und über Agentic AI im Allgemeinen?

---

## Termin 6 – Polish und Generalprobe

**Ziel:** Alles für Termin 7 läuft zuverlässig; die Dokumentation ist vollständig.

- [ ] Repository aufgeräumt: Code formatiert, ungenutzte Dateien entfernt, README aktualisiert, Abhängigkeiten dokumentiert (`requirements.txt` / `pyproject.toml`)
- [ ] **Projekt-Dokumentation (4–8 Seiten)** geschrieben, mit allen Punkten:
  - [ ] Motivation und Zielsetzung
  - [ ] Architekturüberblick (mit Diagramm)
  - [ ] Designentscheidungen und ihre Begründung
  - [ ] Evaluation und Ergebnisse
  - [ ] Limitationen und Failure Modes
  - [ ] Was man mit mehr Zeit anders/zusätzlich machen würde
- [ ] **Live-Demo** vorbereitet (Use Case, benötigte Daten, Notfallplan bei Fehlschlag)
- [ ] **Präsentationsfolien (10–15 Folien)** erstellt – Struktur: Problem & Motivation / Lösungsansatz & Architektur / Demo / Evaluation & Ergebnisse / Erkenntnisse & Limitationen / Fazit
- [ ] **Trockenlauf** in der Gruppe durchgeführt (Zeit gestoppt, alle reden mind. einmal)

**Pflicht-Deliverable Termin 6**

- [ ] Aufgeräumtes Repository mit vollständiger Doku
- [ ] Präsentationsfolien im Repo
- [ ] Backup-Video der Demo (kurzer Bildschirmmitschnitt) für den Notfall

---

## Termin 7 – Abschlusspräsentation

**Format:** 15–20 Min. Vortrag inkl. Live-Demo, danach ca. 10 Min. Diskussion.

- [ ] Vortrag mit Live-Demo nach der in Termin 6 vorbereiteten Struktur gehalten
- [ ] Diskussion bestritten (Fragen zu Designentscheidungen, Trade-offs, Failure Modes)
- [ ] Jedes Gruppenmitglied hat mind. einen inhaltlichen Beitrag geleistet
- [ ] **Übergabe:** finale Dokumentation **und** Repository-Link an den Dozenten

---

## Finale Abgabe-Übersicht (Schnellkontrolle)

Alle nachfolgenden Artefakte sollten am Ende im Repository vorhanden sein:

- [ ] `README.md` (Projektinfo, Gruppe, Architekturskizze – aktuell gehalten)
- [ ] Architekturdokument (1–2 Seiten)
- [ ] Lauffähiges System mit Start-Anleitung
- [ ] Lab Notebook (Beobachtungen aus Durchläufen)
- [ ] Observability-Logs / Logging-Setup
- [ ] Edge-Case-Liste mit Lösungsansätzen
- [ ] Guardrails (sofern projekt-relevant: A, D, F)
- [ ] Metriken-Definition
- [ ] Hypothesen-Dokument mit Versuchsaufbau + Ergebnissen
- [ ] Schriftliche Reflexion
- [ ] Projekt-Dokumentation (4–8 Seiten)
- [ ] Präsentationsfolien
- [ ] Backup-Demo-Video
- [ ] Transparenz-Hinweis zur Nutzung von Coding-Assistenten (wer/was beigetragen hat)

---

