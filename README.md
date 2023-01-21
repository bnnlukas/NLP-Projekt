# Entwicklung eines Chatbots für Themen rund um die Fußball Weltmeisterschaften
Ein Projekt im Fach Aktuelle Data Science Entwicklungen - Natural Language Processing (WWI20DSA) mit den Teilnehmern:
- Lukas
- Aymane 
- Jasmina 
- Pascal 

## Idee
Für die Sportart Fußball soll ein User die Möglichkeit haben etwaige Sieger, teilnehmende Mannschaften oder erzielte Tore etc. anzeigen lassen zu können.  

![Alt text](demo.png)


## Natural Language Processing
### Named Entity Recognition
### Numerizer 
### Intentanalyse
 


## Chatbot
- ~Open-Source Rasa Chatbot (Fragen ob Alternative verwendet werden soll)~ -> eigenes Modell entwickelt und trainiert
- ~APIs zum Anbinden/Integrieren der relevanten Daten~(Nicht umsetzbar, da APIs kosten) -> CSV-Files zu FIFA Weltmeisterschaften als Datenbasis 
- Backend mit Python-Flask
- Frontend mit HTML (CSS) inklusive Javascript Funktion (zur Umsetzung des Chatbot)
- Docker

### Backend mit Python-Flask

### Frontend
### Funktionen des Chatbots
- Visualisierung der Ausgabe
- Interaktion: Kunde/Anwender kann dem Chatbot Fragen stellen welche vom Chatbot beantwortet werden
- Named Entity Recognition: um Vereine (Geopolitical Entity), Ordinalwerte(Platzierungen)und Date erkennen und ausgeben lassen zu können
  - Ersetzen von vordefinierten Synonmen (bsp. Weltmeister = 1. Platz)
- Numerizer: Erkennung von geschriebenen Zahlen (bsp. eins) 
- Chatbot akzeptiert lediglich die englische Sprache
- Intentanalyse: Die Intention hinter der Anfrage des Nutzer verstehen und eine Abfrage diesbezüglich machen 



