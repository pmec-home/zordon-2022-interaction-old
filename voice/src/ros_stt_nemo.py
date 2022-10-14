#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
import pycuda.driver as cuda
from multiprocessing import Process
import numpy as np

from jetson_voice import ASR, AudioInput, AudioWavStream, AudioSamplesStream, ConfigArgParser, list_audio_devices
from soundfile import SoundFile

import numpy as np

import soundfile as sf

import rospy

from voice.srv import voice as voice_srv
from voice.srv import voiceResponse

import logging

class STTNemo:

    def __init__(self, 
                 stt_model="/home/athome/zordon-2022/zordon-2022-interaction/voice/src/data/networks/asr/quartznet-15x5_en/quartznet.beamsearch_lm.json"):
        
        print(f"Creating STT Service")
        service = rospy.Service('zordon/stt/nemo', voice_srv, self) 

        cuda.init()
        self.device = cuda.Device(0) 
        
        print(f"Creating STT model")
        self.stt = ASR(stt_model)

    def __call__(self, req):
        print(req)
        audio_path = req.data

        ctx = self.device.make_context()

        logging.info(f"Starting synthetizing {req.data}")
        
        print(f"Starting streamming on audio {audio_path}")
        stream = AudioWavStream(audio_path,
                                sample_rate=self.stt.sample_rate, 
                                chunk_size=self.stt.chunk_size)

        for samples in stream:
            results = self.stt(samples)
            
            if self.stt.classification:
                print(f"class '{results[0]}' ({results[1]:.3f})")
            else:
                for transcript in results:
                    logging.debug(transcript['text'])
                    if transcript['end']:
                        print("Final sentence:", transcript['text'])
                        with open(f'{audio_path}.txt', 'a') as txt:
                            txt.write(f'{self.__class__.__name__}: {transcript["text"]}')
                        return transcript['text']

        ctx.pop()  
        
        return voiceResponse()

if __name__ == "__main__":
    rospy.init_node('speech_to_text_stt_nemo')
    stt = STTNemo()
    rospy.spin()