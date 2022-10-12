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

import logging

class TTSNemo:

    def __init__(self, 
                 output_device=24,
                 tts_model="fastpitch_hifigan"):
                 
        logging.info(f"Creating tts model")
        self.tts = TTS(tts_model)
        self.audio_device = output_device

        for run in range(args.warmup+1):
            start = time.perf_counter()
            audio = self.tts("warmup")
            stop = time.perf_counter()
            latency = stop-start
            duration = audio.shape[0]/tts.sample_rate
            logging.debug(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
        
    def __call__(self, text):  
        logging.info(f"Starting synthetizing {text}")
        try:
            audio = self.tts("warmup")
            self.audio_device.write(audio)
            return True
        except:
            return False

if __name__ == "__main__":
    tts = TTSNemo()
    
    def handler(req):
        print(req)
        success = tts(req.data)
        return TriggerResponse(
            success=success,
            message=req.data
        )
    rospy.init_node('speech_to_text_tts', anonymous=True)
    service = rospy.Service('zordon/tts', Trigger, handler)    
    
    rospy.spin()