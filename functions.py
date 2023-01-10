import pandas as pd
import re

# spacy imports
import spacy
from spacy.lang.en.examples import sentences 
from numerizer import numerize

replace_dict = {
    'won': 'became the 1st place',
    'winner': 'became the 1st place',
    'world-champion': '1st place',
    'world champion': '1st place',
    'champion ': '1st place',
    'last world championship': 'World Cup in 2022',
    'last world-championship': 'World Cup in 2022',
    'last world cup': 'World Cup in 2022',
    'last world-cup': 'World Cup in 2022',
    'first world championship': 'World Cup in 1930',
    'first world-championship': 'World Cup in 1930',
    'first world cup': 'World Cup in 1930',
    'first world-cup': 'World Cup in 1930'
}

def return_answer(input, intent):
    nlp = spacy.load("en_core_web_sm")
    for key in replace_dict.keys():
        if key in input:
            input = input.replace(str(key), replace_dict[key])
        else:
            pass
    txt = nlp(input)
    try:
        if intent == 'YearHost':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            host = df['HOST'].loc[df['YEAR'] == int(year[0])]
            output = f'The host of the FIFA world cup in {year[0]} was {host.item()}'

        elif intent == 'yearMatches':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            matches = df['MATCHES PLAYED'].loc[df['YEAR'] == int(year[0])]
            output = f'In {year[0]} in total {matches.item()} matches were played'

        elif intent == 'year(avg)Goals':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            avgGoals = df['AVG GOALS PER GAME'].loc[df['YEAR'] == int(year[0])]
            output = f'In {year[0]} in average {avgGoals.item()} goals where scored per game'

        elif intent == 'yearGoals':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            avgGoals = df['GOALS SCORED'].loc[df['YEAR'] == int(year[0])]
            output = f'In {year[0]} in total {avgGoals.item()} goals where scored'

        elif intent == 'PlacementTeam':
            ord_dict = txt._.numerize(labels = 'ORDINAL')
            for key in ord_dict.keys():
                txt_modified = input.replace(str(key), ord_dict[key])
                txt = nlp(txt_modified)
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv(f'./data/FIFA - {int(year[0])}.csv')
            pos_str = [token.lemma_ for token in txt if token.ent_type_ == "ORDINAL"]
            pos_int = int(re.search(r'\d+', pos_str[0]).group())
            pos_team = df['Team'].loc[df['Position'] == pos_int]
            output = f'At the {year[0]} World Cup, the {pos_str[0]} place was taken by {pos_team.item()}'
        
        elif intent == 'TeamPlacement':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv(f'./data/FIFA - {int(year[0])}.csv')
            nation = [token.lemma_ for token in txt if token.ent_type_ == "GPE"]
            team_pos = df['Position'].loc[df['Team'] == nation[0].capitalize()]
            output = f'At the {year[0]} World Cup, {nation[0]} finished the {team_pos.item()}th place'
        return 'Question: ' + input + '\nAnswer: ' + output
    except:
        return 'Please type in your question again'

    

YearHost = return_answer('Where was the world cup 1958?', 'YearHost')
yearMatches = return_answer('How many matches where played in 1930?', 'yearMatches')
avgGoals = return_answer('In average, how many goals where scored in 1998?', 'year(avg)Goals')
Goals = return_answer('How many goals where scored 1950?', 'yearGoals')
PlacementTeam = return_answer('Which Team became the world champion at the first world cup?', 'PlacementTeam')
TeamPlacement = return_answer('Which place did Argentina become in the first world cup?', 'TeamPlacement')

output = return_answer('How many Games were played in 1990?', 'yearMatches')

print(output)