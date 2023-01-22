from intent_analysis import get_intent
from functions import return_answer

bot_name = "Bot"

def get_response(input):
    intent = get_intent(input)
    response = return_answer(input, intent)
    return response
    