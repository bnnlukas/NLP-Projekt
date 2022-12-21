# Aktuelle Data Science Entwicklungen - Natural Language Processing
## Entwicklung eines Chatbots für Themen rund um Sport

### Idee
Für die Männer Sportarten Fußball und Basketball soll ein User die Möglichkeit haben z.b. Spielstände und Ergebnisse, Tabellenplatzierungen und Informationen zu Spielern/Teams und deren Statistiken(Topscorer, Assists, Rebounds etc.) sich per Eingabe bei einem Web Chatbot ausgeben zu lassen. 
Daneben soll die Entwicklung der Platzierungen der Teams, im Verlauf der Saison visuell dargestellt werden.

Mögliche zusätzlichen Implementierungen: Transfermarktinformation inclusive Entwicklung des Wertes eines Spielers

### Umsetzung/Tools
- Open-Source Rasa Chatbot (Fragen ob Alternative verwendet werden soll)
  - gibt schon vordefinierte modelle, man beschäftigt sich mit dem Framework, erklären können wie das funktioniert (NLP Pipeline bearbeiten) 
  - besser selbst modelle trainieren( Zitat: " man macht sich damit das leben leichter")
- APIs zum Anbinden/Integrieren der relevanten Daten
- Backend mit Python-Flask
- Frontend mit HTML (CSS)
- Docker
- Visualisierungen mit Python Packages
- NER & Intentanalyse
- - Sequence tagging


### Teilnehmer
- Lukas  
- Aymane 
- Jasmina 
- Pascal 


### Funktionen des Chatbots
- Visualisierung der Ausgabe
- Interaktion: Kunde/Anwender kann dem Chatbot Fragen stellen welche vom Chatbot beantwortet werden
- Named Entity Recognition



### Useful links
- API Anbindung: https://www.api-football.com/news/post/how-to-use-api-football-with-python


