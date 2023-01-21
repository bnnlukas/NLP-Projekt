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
***„Wie können wir mit unserem Chatbot unseren Usern im Kontext unserer Produkte und der Customer Journeys helfen?“***
- Herausfinden was der User möchte
- Schlagwörter dem eigentlichen Zweck zuordnen
- Auf Basis des Intents, weitere aufgaben des Chatbots anstoßen
- 
**Support Vector Classifier** ist ein Algorithmus für maschinelles Lernen, der für Klassifizierungsaufgaben verwendet wird. Er findet die beste Grenze (Hyperebene), die die Daten in verschiedene Klassen trennt, indem er die Marge (den Abstand zwischen der Grenze und den nächstgelegenen Datenpunkten aus jeder Klasse) maximiert. SVCs können sowohl für lineare als auch für nichtlineare Klassifizierungsaufgaben eingesetzt werden, indem verschiedene Kernel-Funktionen wie lineare, polynomiale und radiale Basisfunktionen (RBF) verwendet werden. Sie gelten als leistungsfähige Klassifikatoren mit guter Genauigkeit, die besonders nützlich sind, wenn die Daten nicht linear trennbar sind und die Anzahl der Merkmale viel größer ist als die Anzahl der Stichproben.

**Word2Vec** ist eine Technik zur Erstellung von Worteinbettungen, d.h. numerischen Darstellungen von Wörtern. Die Idee hinter den Worteinbettungen ist es, Wörter in einem hochdimensionalen Raum so darzustellen, dass semantisch ähnliche Wörter nahe beieinander liegen und unähnliche Wörter weit voneinander entfernt sind. Word2Vec verwendet ein neuronales Netz, um diese Repräsentationen aus einem großen Korpus von Textdaten zu lernen. Die daraus resultierenden Worteinbettungen können für eine Vielzahl von Aufgaben der Verarbeitung natürlicher Sprache verwendet werden.

**TF-IDF** ist eine Technik zur Darstellung der Bedeutung von Wörtern in einem Dokument. Die Idee hinter TF-IDF ist es, Wörter danach zu gewichten, wie häufig sie in einem Dokument vorkommen, wobei auch berücksichtigt wird, wie häufig sie in einem Korpus von Dokumenten vorkommen. Der TF-IDF-Wert eines Wortes ist das Produkt aus seiner Termfrequenz (TF) und seiner inversen Dokumentenfrequenz (IDF). Wörter, die nur in einem bestimmten Dokument vorkommen, haben einen höheren TF-IDF-Wert als Wörter, die in vielen Dokumenten vorkommen.

**gridsearch** ist eine Funktion, die dabei hilft, die beste Kombination von Hyperparametern für ein bestimmtes Modell des maschinellen Lernens zu finden. Dazu wird das Modell mit allen möglichen Kombinationen von Hyperparameterwerten trainiert und die Kombination zurückgegeben, die die beste Punktzahl gemäß der angegebenen Scoring-Metrik ergibt.

**k-fold**  ist eine Technik zur Bewertung der Leistung eines maschinellen Lernmodells, bei der die Daten in k Teilmengen aufgeteilt werden, das Modell auf k-1 der Teilmengen trainiert und auf der verbleibenden Teilmenge bewertet wird. Der Prozess wird k-mal wiederholt und die Leistung wird über alle k Iterationen gemittelt.



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


### Quellen
- https://www.atlantis-press.com/journals/hcis/125963694
