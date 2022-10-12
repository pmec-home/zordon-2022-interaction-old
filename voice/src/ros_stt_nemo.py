#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from jetson_voice import ASR, AudioInput, AudioWavStream, AudioSamplesStream, ConfigArgParser, list_audio_devices
from soundfile import SoundFile
import soundfile as sf

import rospy

from std_srvs.srv import Trigger, TriggerResponse

import logging

class STTNemo:

    def __init__(self, 
                 stt_model="quartznet"):
                 
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
                        with open(f'{audio_path}.txt', 'w') as txt:
                            txt.write(transcript['text'])
                        return transcript['text']

        return None

if __name__ == "__main__":
    stt = STTNemo()
    
    def handler(req):
        print(req)
        transcription = stt(req.data)
        return TriggerResponse(
            success=True if transcription is not None else False,
            message=transcription
        )
    rospy.init_node('speech_to_text_stt_nemo', anonymous=True)
    service = rospy.Service('zordon/stt/nemo', Trigger, handler)    
    
    rospy.spin()