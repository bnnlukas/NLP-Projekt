# Entwicklung eines Chatbots für Themen rund um die Fußball Weltmeisterschaften
## Aktuelle Data Science Entwicklungen - Natural Language Processing (WWI20DSA)

### Idee
Für die Sportart Fußball soll ein User die Möglichkeit haben etwaige Sieger, teilnehemende Mannschaften oder erzielte Tore etc. anzeigen lassen zu können.  

![Alt text](demo.png)


### Umsetzung/Tools
- ~Open-Source Rasa Chatbot (Fragen ob Alternative verwendet werden soll)~ -> eigenes Modell entwickelt und trainiert
- ~APIs zum Anbinden/Integrieren der relevanten Daten~(Nicht umsetzbar, da APIs kosten) -> CSV-Files zu FIFA Weltmeisterschaften als Datenbasis 
- Backend mit Python-Flask
- Frontend mit HTML (CSS) inklusive Javascript Funktion (zur Umsetzung des Chatbot)
- Docker
- NER & Intentanalyse



### Funktionen des Chatbots
- Visualisierung der Ausgabe
- Interaktion: Kunde/Anwender kann dem Chatbot Fragen stellen welche vom Chatbot beantwortet werden
- Named Entity Recognition: um Vereine (Geopolitical Entity), Ordinalwerte(Platzierungen)und Date erkennen und ausgeben lassen zu können
  - Ersetzen von vordefinierten Synonmen (bsp. Weltmeister = 1. Platz)
- Numerizer: Erkennung von geschriebenen Zahlen (bsp. eins) 
- Chatbot akzeptiert lediglich die englische Sprache
- Intentanalyse: Die Intention hinter der Anfrage des Nutzer verstehen und eine Abfrage diesbezüglich machen

### Teilnehmer
- Lukas
- Aymane 
- Jasmina 
- Pascal 
