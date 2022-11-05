#!/usr/bin/env python3
# coding: utf-8

import re
import sys
import time
import pyloudnorm as pyln
import pycuda.driver as cuda
import numpy as np

from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile
import soundfile as sf
from num2words import num2words
import rospy

from voice.srv import tts as tts_srv
from voice.srv import ttsResponse

import logging

class TTSNemo:

    def __init__(self, 
                 output_device=24,
                 warmup=5,
                 normalize=True,
                 tts_model="/home/athome/zordon-2022/zordon-2022-interaction/voice/src/data/networks/tts/fastpitch_hifigan/fastpitch_hifigan.json"):
        self.normalize = normalize
        logging.info(f"Creating tts model")
        service = rospy.Service('zordon/tts', tts_srv, self) 
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

    def _normalize(self, text):
        print("Normalizing", text)
        text = text.lower()
        for c in 'ãáàâ':
            if c in text:
                text = text.replace(c, 'a')
        for c in 'ç':
            if c in text:
                text = text.replace(c, 'ç')
        for c in 'éê':
            if c in text:
                text = text.replace(c, 'e')
        for c in 'í':
            if c in text:
                text = text.replace(c, 'i')
        for c in 'õóô':
            if c in text:
                text = text.replace(c, 'o')
        for c in 'úü':
            if c in text:
                text = text.replace(c, 'u')
        words = []
        for word in text.split():
            if word.isdigit():
                word = num2words(word)
            words.append(word)
        text = ' '.join(words)
        text = re.sub('[^a-zA-Z\s]', "", text)
        print("Normalized text:", text)
        return text

    def __call__(self, req):
        print(req)

        logging.info(f"Starting synthetizing {req.text}")
        input = req.text
        if self.normalize:
            input = self._normalize(input)

        try:
            # Gabiarra:
            ctx = self.device.make_context()  
            audio = self.tts("input")
            if round(sum(audio)/len(audio), 1) == 0.0:
            	ctx = self.device.make_context()
            	try:
                    audio = self.tts(input)
            	except: pass
            	ctx.pop()  
            ctx.pop()  
        
            audio_device = AudioOutput(self.output_device, self.tts.sample_rate)
            audio_device.write(audio)
            
        except Exception as e:
            print("ERROR tts: {e}")

        
        return ttsResponse()

if __name__ == "__main__":
    rospy.init_node('text_to_speech_nemo')
    tts = TTSNemo()
    rospy.spin()
