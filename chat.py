from intent_detection import get_intent
from functions import return_answer

bot_name = "Bot"

# Definieren der Funktion, welche für einen gegebenen Eingabetext eine Antwort bestimmt und zurückgibt
def get_response(input):
    intent = get_intent(input)
    response = return_answer(input, intent)
    return response
    