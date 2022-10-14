#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import time
import pyloudnorm as pyln
import numpy as np

from jetson_voice import ASR, AudioInput, AudioWavStream, AudioSamplesStream, ConfigArgParser, list_audio_devices
from soundfile import SoundFile
import soundfile as sf

import rospy

from std_srvs.srv import Trigger, TriggerResponse

import logging

class VADRecorder:

    def __init__(self, 
                 vad_model="/home/athome/zordon-2022/zordon-2022-interaction/voice/src/data/networks/asr/vad_marblenet/vad_marblenet.json",
                 max_samples=16_000*10, 
                 normalize=True,
                 mic=24,
                 target_norm_lufs=-12.0,
                 background_detection_patience=15):
        
        logging.info(f"Creating VAD model")
        self.vad = ASR(vad_model)
        self.background_detection_patience = background_detection_patience
        self.normalize = normalize
        self.max_samples = max_samples
        self.background_detection_patience = background_detection_patience
        self.target_norm_lufs = target_norm_lufs 
        logging.info(f"Starting mic streamming on device {mic}")
        self.stream = AudioInput(mic=mic, 
                                 sample_rate=self.vad.sample_rate, 
                                 chunk_size=self.vad.chunk_size)
        
    def __call__(self, req, prefix=''): 
        print(req)
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        background_detection_patience = self.background_detection_patience

        final_samples = []
        while background_detection_patience > 0:
            samples = next(self.stream)
            vad_results = self.vad(samples)
            logging.info(str(vad_results) + " | background_detection_patience: " + 
                         str(background_detection_patience))
            if vad_results[0] == 'speech' and vad_results[1] > 0.9:
                background_detection_patience = self.background_detection_patience
            else:
                background_detection_patience -= 1
            final_samples = [*final_samples, *samples]
            if len(final_samples) > self.max_samples:
                logging.info('Stop recording (max samples reached)')
                break

        logging.info('Closing stream')
        self.stream.close()
        print('\naudio stream closed.')

        audio_name = prefix+current_time+".wav"

        if self.normalize:
            output_wav = SoundFile(audio_name, mode='w', samplerate=16_000, channels=1)
            output_wav.write(final_samples)
            final_samples = np.array(final_samples)
            
            logging.debug('Audio shape:', final_samples.shape)

            if not self.normalize: 
                return TriggerResponse(
                    success=True,
                    message=os.path.abspath(audio_name)
                ) 

            audio_name = prefix+current_time+".norm.wav"

            data, rate = sf.read(current_time+".wav") # load audio
            # peak normalize audio to -1 dB
            peak_normalized_audio = pyln.normalize.peak(data, -1.0)

            # measure the loudness first 
            meter = pyln.Meter(rate) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)

            # loudness normalize audio to -12 dB LUFS
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.target_norm_lufs)

            output_wav_norm = SoundFile(audio_name, mode='w', samplerate=16_000, channels=1)
            output_wav_norm.write(loudness_normalized_audio)

        return TriggerResponse(
            success=True,
            message=os.path.abspath(audio_name)
        ) 

if __name__ == "__main__":

    vad_recorder = VADRecorder()

    rospy.init_node('speech_to_text_vad', anonymous=True)
    service = rospy.Service('zordon/vad', Trigger, vad_recorder)    
    
    rospy.spin()