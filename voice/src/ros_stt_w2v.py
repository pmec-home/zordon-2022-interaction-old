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

import rospy

from std_srvs.srv import Trigger, TriggerResponse

import logging

class STTW2V:

    def __init__(self, model_name="jonatasgrosman/wav2vec2-xls-r-1b-english"):
        logging.info(f"Creating STT model")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            logging.error("Error detecting cuda device!")
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

        self.hotword_weight = 10
        self.hotwords = ["zordon", "follow", "me", "often", "creator", "favorite", 
            "is", "in", "hitchbot", "ever", "are", "allowed", "it's", 
            "style", "what", "for", "movies", "away", "mark", "ar", "di" 
            "python", "call", "music", "you", "has", "was", "do", "safe", "run", 
            "get", "self", "driving", "cars", "why", "who", "robots", "zuckerberg", 
            "language", "lunch", "be", "invented", "killed", "compiler", "kind", "ate", 
            "your", "like", "salad", "created", "did", "of", "programming", "robot", 
            "shouldn't", "person", "so", "the", "angry", "yes", "no", "up", "down", "left", "we", 
            "on", "off", "to", "go"]
        logging.info(f"Hotwords is {self.hotwords}")
        
    def __call__(self, audio_path, **kwargs):  
        kwargs.set_default("hotwords", self.hotwords)
        kwargs.set_default("hotwords_weight", self.hotwords_weight)
        logging.info(f"Starting transcribing audio {audio_path}")
        
        try:
            audio, sr = torchaudio.load(audio_path)
            audio = torch.mean(audio, dim=0)
            result = self.stt(audio.numpy(), **kwargs)
            with open(f'{audio_path}.txt', 'w') as txt:
                txt.write(result["text"])
        except Exception as e:
            logging.error(f"Trascribing {audio_path}: {str(e)}")
            return None

        return result["text"]

if __name__ == "__main__":
    stt = STTW2V()
    
    def handler(req):
        print(req)
        transcription = stt(req.data)
        return TriggerResponse(
            success=True if transcription is not None else False,
            message=transcription
        )
    rospy.init_node('speech_to_text_stt_w2v', anonymous=True)
    service = rospy.Service('zordon/stt/w2v', Trigger, handler)    
    
    rospy.spin()