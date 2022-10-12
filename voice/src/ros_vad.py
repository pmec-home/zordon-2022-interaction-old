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

class VADRecorder:

    def __init__(self, 
                 vad_model="marblenet",
                 max_samples=16_000*10, 
                 normalize=True,
                 mic=24,
                 target_norm_lufs=-12.0,
                 background_detection_patience=15):
                 
        logging.info(f"Creating VAD model")
        self.vad = ASR(vad_model)
        self.normalize = normalize
        self.max_samples = max_samples
        self.background_detection_patience = background_detection_patience
        self.target_norm_lufs = target_norm_lufs 
        logging.info(f"Starting mic streamming on device {mic}")
        self.stream = AudioInput(mic=mic, 
                                 sample_rate=vad.sample_rate, 
                                 chunk_size=vad.chunk_size)
        
    def __call__(self): 
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
        background_detection_tresh = self.background_detection_tresh

        final_samples = []
        while background_detection_tresh > 0:
            samples = next(stream)
            vad_results = self.vad(samples)
            logging.info(str(vad_results) + " | background_detection_tresh: " + 
                         str(background_detection_tresh))
            if vad_results[0] == 'speech' and vad_results[1] > 0.9:
                background_detection_tresh = self.background_detection_tresh
            else:
                background_detection_tresh -= 1
            asr_samples = [*asr_samples, *samples]
            if len(asr_samples) > MAX_SAMPLES:
                logging.info('Stop recording (max samples reached)')
                break

        logging.info('Closing stream')
        stream.close()
        print('\naudio stream closed.')

        audio_path = current_time+".wav"

        if self.normalize:
            output_wav = SoundFile(audio_path, mode='w', samplerate=16_000, channels=1)
            output_wav.write(asr_samples)
            asr_samples = np.array(asr_samples)
            
            logging.debug('Audio shape:', asr_samples.shape)

            if not self.normalize: return audio_path
            audio_path = current_time+".norm.wav"

            data, rate = sf.read(current_time+".wav") # load audio
            # peak normalize audio to -1 dB
            peak_normalized_audio = pyln.normalize.peak(data, -1.0)

            # measure the loudness first 
            meter = pyln.Meter(rate) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)

            # loudness normalize audio to -12 dB LUFS
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.target_norm_lufs)

            output_wav_norm = SoundFile(audio_path, mode='w', samplerate=16_000, channels=1)
            output_wav_norm.write(loudness_normalized_audio)

        return audio_path

if __name__ == "__main__":
    vad_recorder = VADRecorder()
    
    def handler(req):
        print(req)
        audio_path = vad_recorder()
        return TriggerResponse(
            success=True,
            message=audio_path
        )
    rospy.init_node('speech_to_text_vad', anonymous=True)
    service = rospy.Service('zordon/vad', Trigger, handler)    
    
    rospy.spin()