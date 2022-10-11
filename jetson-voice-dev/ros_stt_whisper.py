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

from std_srvs.srv import Trigger, TriggerResponse

import logging

class STTWhisper:

    def __init__(self, model_name="base"):
        self.stt = whisper.load_model("medium")
        self.decoding_options = whisper.DecodingOptions(
            language="en", without_timestamps=True, beam_size=1)
        
    def __call__(self, audio_path, **kwargs):
        kwargs.set_default("decoding_options", self.decoding_options)
        logging.info(f"Starting transcribing audio {audio_path}")
        try:
            audio = self.stt.pad_or_trim(audio.flatten())
            audio = torch.from_numpy(audio).to("cuda")
            
            st = time.time()
            mel = self.stt.log_mel_spectrogram(audio)
            result = self.stt.decode(mel, **kwargs)
            with open(f'{audio_path}.txt', 'w') as txt:
                txt.write(result.text)
            logging.debug(f"{result}")
            end = time.time()
            logging.debug(f"\ttook {end-st} seconds")
        except Exception as e:
            logging.error(f"Trascribing {audio_path}: {str(e)}")
            return None

        return result.text

if __name__ == "__main__":
    stt = STTWhisper()
    
    def handler(req):
        print(req)
        transcription = stt(req.data)
        return TriggerResponse(
            success=True if transcription is not None else False,
            message=transcription
        )
    rospy.init_node('speech_to_text_stt_whisper', anonymous=True)
    service = rospy.Service('zordon/stt/whisper', Trigger, handler)    
    
    rospy.spin()