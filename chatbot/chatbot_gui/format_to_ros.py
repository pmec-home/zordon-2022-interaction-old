
def get_intent(bot_response):
    intent = bot_response["intent"]["name"]
    return intent

def get_entity(bot_response):
    entities = []
    for entity in bot_response['entities']:
        entities.append({"entity":entity["entity"], "value":entity["value"]})
    return entities

def get_bot_answer(bot_response):
    bot_answer = bot_response[0]["text"]
    return bot_answer
