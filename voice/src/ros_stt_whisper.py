#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from soundfile import SoundFile
import soundfile as sf
import torch
import torchaudio
import whisper

import rospy

from voice.srv import stt as stt_srv
from voice.srv import sttResponse

class STTWhisper:

    def __init__(self, model_name="base"):
        self.stt = whisper.load_model(model_name)
        self.decoding_options = whisper.DecodingOptions(
            language="en", without_timestamps=True, beam_size=1)
        # warmup
        mel = whisper.log_mel_spectrogram(torch.empty(torch.Size([480000])).to("cuda"))
        self.stt.decode(mel, self.decoding_options)
        
    def __call__(self, audio_path, decoding_options=None):
        if not decoding_options:
            decoding_options = self.decoding_options
        print(f"Starting transcribing audio {audio_path}")
        try:
            audio = whisper.load_audio(file=audio_path, sr=16_000)
            audio = whisper.pad_or_trim(audio.flatten())
            audio = torch.from_numpy(audio).to("cuda")
            print(audio.shape)
            
            st = time.time()
            mel = whisper.log_mel_spectrogram(audio)
            result = self.stt.decode(mel, decoding_options)
            with open(f'{audio_path}.txt', 'a') as txt:
                txt.write(f'{self.__class__.__name__}: {result.text}')
            print(f"{result}")
            end = time.time()
            print(f"\ttook {end-st} seconds")
        except Exception as e:
            print(f"Trascribing {audio_path}: {str(e)}")
            raise e
            return None

        return result.text

if __name__ == "__main__":
    stt = STTWhisper()
    
    def handler(req):
        print(req)
        transcription = stt(req.audio_path)
        print(transcription)
        return sttResponse(
            transcription=transcription
        )
    rospy.init_node('speech_to_text_stt_whisper', anonymous=True)
    service = rospy.Service('zordon/stt/whisper', stt_srv, handler)   
    
    rospy.spin()