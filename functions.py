import pandas as pd
import re

# spacy imports
import spacy
from spacy.lang.en.examples import sentences 
from numerizer import numerize

replace_dict = {
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
    input = input.lower()
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
            try:
                host = df['HOST'].loc[df['YEAR'] == int(year[0])]
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            output = f'The host of the FIFA World Cup in {year[0]} was {host.item()}.'

        elif intent == 'yearMatches':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            try:
                matches = df['MATCHES PLAYED'].loc[df['YEAR'] == int(year[0])]
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            output = f'In {year[0]} a total of {matches.item()} matches were played.'

        elif intent == 'year(avg)Goals':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            try:
                avgGoals = df['AVG GOALS PER GAME'].loc[df['YEAR'] == int(year[0])]
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            output = f'In {year[0]} on average {avgGoals.item()} goals were scored per game.'

        elif intent == 'yearGoals':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            df = pd.read_csv('./data/FIFA - World Cup Summary.csv')
            try:
                avgGoals = df['GOALS SCORED'].loc[df['YEAR'] == int(year[0])]
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            output = f'In {year[0]} in total {avgGoals.item()} goals were scored.'

        elif intent == 'PlacementTeam':
            ord_dict = txt._.numerize(labels = 'ORDINAL')
            for key in ord_dict.keys():
                txt_modified = input.replace(str(key), ord_dict[key])
                txt = nlp(txt_modified)
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            try:
                df = pd.read_csv(f'./data/FIFA - {int(year[0])}.csv')
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            pos_str = [token.lemma_ for token in txt if token.ent_type_ == "ORDINAL"]
            pos_int = int(re.search(r'\d+', pos_str[0]).group())
            pos_team = df['Team'].loc[df['Position'] == pos_int]
            output = f'At the World Cup in {year[0]}, the {pos_str[0]} place was taken by {pos_team.item()}.'
        
        elif intent == 'firstPlace':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            try:
                df = pd.read_csv(f'./data/FIFA - {int(year[0])}.csv')
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            pos_team = df['Team'].loc[df['Position'] == 1]
            output = f'At the World Cup in {year[0]}, {pos_team.item()} became the World Champion.'
        
        elif intent == 'TeamPlacement':
            year = [token.lemma_ for token in txt if token.ent_type_ == "DATE"]
            try:
                df = pd.read_csv(f'./data/FIFA - {int(year[0])}.csv')
            except:
                output = 'In this year no World Cup was played. Please type in a correct year'
                return output
            nation = [token.lemma_ for token in txt if token.ent_type_ == "GPE"]
            if len(nation) > 1:
                nation = [" ".join(nation)]
            else:
                pass
            team_pos = df['Position'].loc[df['Team'] == nation[0].title()]
            output = f'In the {year[0]} World Cup, {nation[0].title()} finished in the {team_pos.item()}th place'

        elif intent == 'greeting':
            output = f"""Hi, welcome to the FIFA World Cup Chatbot! I was created in the Natural Language 
            Processing lecture of WWI20DSA by Pascal, Lukas, Jasmina and Aymane. I will answer your 
            questions regarding all the FIFA World Cups as best as I can. Do you have a question already?
            I am happy to help you out :)"""

        elif intent == 'thankYou':
            output = f'You are welcome! Do you have any additional questions? To get to our easter egg for some fun, you can ask me who the best player of all time is!'

        elif intent == 'bye':
            output = f'Good bye, it was nice chatting with you. See you again soon!'

        return output
    except:
        return 'I am sorry, but I cannot answer your question. Could you rephrase it for me please?'