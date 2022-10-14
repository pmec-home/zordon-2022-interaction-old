#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
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
        logging.info(f"Creating STT model")
        self.stt = ASR(stt_model)
        
    def __call__(self, audio_path):  
        logging.info(f"Starting streamming on audio {audio_path}")
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
        
        return None

if __name__ == "__main__":
    stt = STTNemo()
    
    def handler(req):
        print(req)
        transcription = stt(req.data)
        print(transcription)
        return voiceResponse()
    rospy.init_node('speech_to_text_stt_nemo', anonymous=True)
    service = rospy.Service('zordon/stt/nemo', voice_srv, handler)    
    
    rospy.spin()