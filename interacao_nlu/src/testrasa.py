from rasa_nlu.model import Interpreter

interpreter = Interpreter.load("./rasa/models/nlu")
result = interpreter.parse("Hey zordon can u hear me?")
print(result)
