#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile
import soundfile as sf

import rospy

from std_srvs.srv import Trigger, TriggerResponse   
from std_msgs.msg import String
from voice.srv import voice as voice_srv
from voice.srv import voiceResponse

import logging

class TTSNemo:

    def __init__(self, 
                 output_device=24,
                 warmup=5,
                 tts_model="/home/athome/zordon-2022/zordon-2022-interaction/voice/src/data/networks/tts/fastpitch_hifigan/fastpitch_hifigan.json"):
                 
        logging.info(f"Creating tts model")
        self.tts = TTS(tts_model)
        self.audio_device = AudioOutput(output_device, self.tts.sample_rate)

        for run in range(warmup+1):
            start = time.perf_counter()
            audio = self.tts("warmup")
            stop = time.perf_counter()
            latency = stop-start
            duration = audio.shape[0]/self.tts.sample_rate
            logging.debug(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
        
        audio = self.tts("Hello, I am Zordon")
        self.audio_device.write(audio)

    def __call__(self, text):  
        logging.info(f"Starting synthetizing {text}")
        try:
            audio = self.tts(text)
            self.audio_device.write(audio)
            return True
        except:
            return False

if __name__ == "__main__":
    tts = TTSNemo()
    def handler(req):
        print(req)
        success = tts(req.data)
        return voiceResponse()
    rospy.init_node('text_to_speech_nemo')
    service = rospy.Service('zordon/tts', voice_srv, handler) 
    # rospy.Subscriber("tts", String, handler)
    
    rospy.spin()