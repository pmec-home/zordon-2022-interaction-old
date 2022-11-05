#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import requests
import json
import pandas as pd
import spacy
import os
import re
from datetime import datetime
import time

directory = os.path.dirname(os.path.realpath(__file__))
log_dir = "/home/athome/zordon-2022/zordon-2022-interaction/chatbot/logs"

import sys
current_module = sys.modules[__name__]

import rospy
from std_srvs.srv import Empty, Trigger
from voice.srv import stt as stt_srv
from voice.srv import tts as tts_srv
from voice.srv import vad as vad_srv

from interacao_nlu.srv import Nlu

###########################################################
# List of Intens
# AWALYS CHANGE THIS LIST WHEN ,ADD A NEW INTENT, OR CHANGE AN INTENT NAME
##########################################################
from enum import Enum
class Intents(Enum):
    MOVE = 'move'
    GREET = 'greet'
    QUESTION = 'question'
    FOLLOW = 'follow'
    AFFIRM = "affirm"
    DENY = "deny"
    START = "start"
    STOP = "stop"
    PICK = "pick"
    BYE = "bye"
    NONE = None
    

###########################################################
# Initilize some global variables
# the list of questions and answer and its nlp abstraction
##########################################################
#Questions and Answer dataset
QandA = pd.read_csv(directory+'/resources/questions_and_answers.csv', sep='|')
nlp = spacy.load('en_core_web_sm')
def load_nlp(word_list):
    global nlp
    nlp_list = []
    for word in word_list:
        print(word)
        nlp_list.append(nlp(str(word)))
    return nlp_list

def compareToNlpList(phrase, nlp_list):
    global nlp
    nlp_phrase = nlp(phrase)
    ranks = []
    for element in nlp_list:
        ranks.append({})
        ranks[-1]['text'] = element.text
        ranks[-1]['similarity'] = nlp_phrase.similarity(element)
    return sorted(ranks, key=lambda x: x['similarity'], reverse=True)
nlp_list = load_nlp(QandA['QUESTION'])
#The database


class _Chatbot():
    def __init__(self):
        # define the name of the action which can then be included in training stories
        x = "sup"

    def read(self, message):
        global nlp_list, QandA
        rank = compareToNlpList(message, nlp_list)
        if(float(rank[0]['similarity']) > 0.6):
            #Grab answer form the Q and A dataframe
            answer = QandA[QandA['QUESTION'] == rank[0]['text']]['ANSWER'].iloc[0]
            #If the answer is in the format ${code} grab the code inside and run it
            # if '$' in answer:
            #     code = re.search('\${(.*)}', answer).group(1)
            #     answer = eval(code)
            return answer
        return "Sorry I did not understand your question"

class ChatBot():
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        self.log_file = f'{log_dir}/{dt_string}.log'
        print('Saving log file', self.log_file)

        print('init node')
        rospy.init_node("chatbot")
        print("wait_for_service('/zordon/wake_word')")
        rospy.wait_for_service('/zordon/wake_word')
        print("wait_for_service('/zordon/vad')")
        rospy.wait_for_service('/zordon/vad')
        print("wait_for_service('/zordon/tts')")
        rospy.wait_for_service('/zordon/tts')
        print("wait_for_service('/zordon/whisper')")
        rospy.wait_for_service('/zordon/stt/whisper')
        # rospy.wait_for_service('/zordon/stt/w2v')
        # print("wait_for_service('/zordon/nlu')")
        # rospy.wait_for_service('/zordon/nlu')	

        print("========================================")

        self.vad = rospy.ServiceProxy('/zordon/vad', vad_srv)
        self.tts = rospy.ServiceProxy('/zordon/tts', tts_srv)
        self.stt = rospy.ServiceProxy('/zordon/stt/whisper', stt_srv)
        self.wake_word = rospy.ServiceProxy('/zordon/wake_word', Empty)
        # self.nlu = rospy.ServiceProxy('/zordon/nlu', Nlu)
        self.chatbot = _Chatbot()

    def _write_log(self, msg):
        with open(self.log_file, 'a') as log:
            print(datetime.now(), msg, file=log)

    def listen(self):
        self._write_log('='*60)
        print('Waiting wake word...')
        self._write_log('Waiting wake word')
        self.wake_word()
        self._write_log('Recording voice')
        print('Listening command...')
        vad_response = self.vad()
        print(vad_response)
        self._write_log(f'Voice recorded: {vad_response.audio_path}')
        os.system(f'cp {vad_response.audio_path} {log_dir}')
        stt_response = self.stt(vad_response.audio_path)
        self._write_log(f'Transcription: {stt_response.transcription}')
        print(stt_response)
        #tts_response = self.tts(stt_response.transcription)
        # nlu_response = self.nlu(stt_response.transcription)
        # print(nlu_response)
        answer = self.chatbot.read(stt_response.transcription)
        self._write_log(f'Answer: {answer}')
        self.tts(answer[:100])



if __name__ == "__main__":
    print("Creating ChatBot")
    chatbot = ChatBot(log_dir)
    while(True):
        chatbot.listen()
