
# from rasa.core.agent import Agent
from chat_gui import GUI
import format_to_ros as ftr
import warnings
import asyncio

warnings.filterwarnings("ignore")
agent = None#Agent.load("rasa/models/20221016-044557-tranquil-kiosk.tar.gz")
stt_string = ""

# TODO: choose title name and dimensions
main_gui = GUI(rasa_agent=agent, title="Title", dimension="400x500",
                exit_msg="turn off the beat")
main_gui.start()
