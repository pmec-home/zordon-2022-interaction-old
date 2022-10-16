#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from soundfile import SoundFile
import soundfile as sf
import torchaudio
import torch
from transformers import (
    AutoTokenizer, 
    AutoFeatureExtractor, 
    AutomaticSpeechRecognitionPipeline, 
    Wav2Vec2ForCTC
)

import rospy

from voice.srv import stt as stt_srv
from voice.srv import sttResponse

import logging

class STTW2V:

    def __init__(self, model_name="jonatasgrosman/wav2vec2-xls-r-1b-english"):
        print(f"Creating STT model")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            print("Error detecting cuda device!")
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        if device >= 0:
            model = model.to(f"cuda:{device}")
        self.stt = AutomaticSpeechRecognitionPipeline(
            feature_extractor=feature_extractor, 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            chunk_length_s=5, 
            stride_length_s=(4, 2))

        self.hotwords_weight = 10
        self.hotwords = ["zordon", "follow", "me", "often", "creator", "favorite", 
            "is", "in", "hitchbot", "ever", "are", "allowed", "it's", 
            "style", "what", "for", "movies", "away", "mark", "ar", "di" 
            "python", "call", "music", "you", "has", "was", "do", "safe", "run", 
            "get", "self", "driving", "cars", "why", "who", "robots", "zuckerberg", 
            "language", "lunch", "be", "invented", "killed", "compiler", "kind", "ate", 
            "your", "like", "salad", "created", "did", "of", "programming", "robot", 
            "shouldn't", "person", "so", "the", "angry", "yes", "no", "up", "down", "left", "we", 
            "on", "off", "to", "go"]
        print(f"Hotwords is {self.hotwords}")
        
    def __call__(self, audio_path, **kwargs):  
        kwargs.setdefault("hotwords", self.hotwords)
        kwargs.setdefault("hotwords_weight", self.hotwords_weight)
        print(f"Starting transcribing audio {audio_path}")
        
        try:
            audio, sr = torchaudio.load(audio_path)
            audio = torch.mean(audio, dim=0)
            result = self.stt(audio.numpy(), **kwargs)
            with open(f'{audio_path}.txt', 'a') as txt:
                txt.write(f'{self.__class__.__name__}: {result["text"]}')
        except Exception as e:
            print(f"Trascribing {audio_path}: {str(e)}")
            return None

        return result["text"]

if __name__ == "__main__":
    stt = STTW2V()
    
    def handler(req):
        print(req)
        transcription = stt(req.audio_path)
        print(transcription)
        return sttResponse(
            transcription=transcription
        )
    rospy.init_node('speech_to_text_stt_w2v', anonymous=True)
    service = rospy.Service('zordon/stt/w2v', stt_srv, handler)    
    
    rospy.spin()