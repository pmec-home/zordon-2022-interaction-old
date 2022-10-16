#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import time
import pyloudnorm as pyln
import numpy as np

from jetson_voice import AudioInput, AudioWavStream
from jetson_voice.utils import audio_to_float
from soundfile import SoundFile
import soundfile as sf

import rospy
import torch
torch.set_num_threads(1)

from pprint import pprint
from voice.srv import vad as vad_srv
from voice.srv import vadResponse

import logging

class VADRecorder:

    def __init__(self, 
                 max_samples=16_000*6, 
                 normalize=True,
                 mic=24,
                 target_norm_lufs=-12.0,
                 warmup=5,
                 audio_prefix='',
                 chunk_size=4_000,
                 background_detection_patience=(16_000*4)//4_000):
        
        self.mic = mic
        self.prefix = audio_prefix

        print(f"Creating service zordon/vad")
        service = rospy.Service('zordon/vad', vad_srv, self) 

        
        print(f"Creating VAD model")
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)
        self.model.to("cuda")
        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = self.utils
        print(f"Creating VAD model: DONE")
        self.sample_rate = 16_000
        self.chunk_size = chunk_size

        self.vad_iterator = VADIterator(self.model)

        print(f"Testing mic streaming on device {self.mic}")
        try:
            stream = AudioInput(mic=self.mic, 
                                sample_rate=self.sample_rate, 
                                chunk_size=self.chunk_size)
        except Exception as e:
            print(f"ERROR creating streaming on device {self.mic}")
        for i in range(5):
            prob = self.model(torch.rand((1,16_000), dtype=torch.float32).to("cuda"), 16_000).item()
            print(prob)
            
        self.background_detection_patience = background_detection_patience
        self.normalize = normalize
        self.max_samples = max_samples
        self.background_detection_patience = background_detection_patience
        self.target_norm_lufs = target_norm_lufs 

    def int2float(sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound
        
    def __call__(self, req):
        prob = self.model(torch.rand((1,16_000), dtype=torch.float32).to("cuda"), 16_000).item()
        os.system(f'aplay /home/athome/zordon-2022/zordon-2022-interaction/voice/resources/okay2.wav -D hw:2,0')
        print(req)
        
        print(f"Starting mic streaming on device {self.mic}")
        try:
            stream = AudioInput(mic=self.mic, 
                                sample_rate=self.sample_rate, 
                                chunk_size=self.chunk_size)
        except Exception as e:
            print(f"ERROR creating streaming on device {self.mic}")
            return None

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        background_detection_patience = self.background_detection_patience

        final_samples = []
        all_samples = []

        voice_detected = False
        while background_detection_patience > 0:
            samples = next(stream)
            if len(samples) < self.chunk_size:
                break
            prob = self.model(torch.tensor(VADRecorder.int2float(samples)).to("cuda"), 16_000).item()
            print(prob)

            if prob > 0.9:
                background_detection_patience = self.background_detection_patience
                print("Detected voice")
                voice_detected = True
            else:
                background_detection_patience -= 1
                print(background_detection_patience)

            if voice_detected:
                final_samples = [*final_samples, *samples]
            all_samples = [*all_samples, *samples]

            if len(final_samples) > self.max_samples: 
                break
            
        self.vad_iterator.reset_states() 

        if len(final_samples) == 0:
            final_samples = all_samples
        
        print('Closing stream')
        stream.close()
        print('\naudio stream closed.')

        audio_name = self.prefix+current_time+".wav"
        output_wav = SoundFile(audio_name, mode='w', samplerate=16_000, channels=1)
        output_wav.write(final_samples)

        final_samples = np.array(final_samples)
        
        print('Audio shape:', final_samples.shape)

        if self.normalize:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

            audio_name = self.prefix+current_time+".norm.wav"

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

        os.system(f'aplay /home/athome/zordon-2022/zordon-2022-interaction/voice/resources/activate.wav -D hw:2,0')
                    
        print(f"Saving audio {os.path.abspath(audio_name)}")
        return vadResponse(
            audio_path=os.path.abspath(audio_name)
        )


if __name__ == "__main__":
    rospy.init_node('speech_to_text_vad')
    vad_recorder = VADRecorder()
    rospy.spin()