#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
import pycuda.driver as cuda
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
        service = rospy.Service('zordon/tts', voice_srv, self) 
        self.output_device = output_device
        cuda.init()
        self.device = cuda.Device(0) 

        self.tts = TTS(tts_model)

        for run in range(warmup+1):
            start = time.perf_counter()
            audio = self.tts("warmup")
            stop = time.perf_counter()
            latency = stop-start
            duration = audio.shape[0]/self.tts.sample_rate
            logging.debug(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
        
        try:
            audio_device = AudioOutput(self.output_device, self.tts.sample_rate)
            audio = self.tts("Hello, I am Zordon")
            audio_device.write(audio)
        except Exception as e:
            print("ERROR", str(e))

    def __call__(self, req):
        print(req)

        ctx = self.device.make_context()

        logging.info(f"Starting synthetizing {req.data}")
        audio = self.tts(req.data)
        audio_device = AudioOutput(self.output_device, self.tts.sample_rate)
        audio_device.write(audio)

        ctx.pop()  
        
        return voiceResponse()

if __name__ == "__main__":
    rospy.init_node('text_to_speech_nemo')
    tts = TTSNemo()
    rospy.spin()