#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from jetson_voice import ASR, AudioInput, ConfigArgParser, list_audio_devices
from soundfile import SoundFile
import soundfile as sf
import torch, torchaudio
from os import getenv
from fastapi import FastAPI, HTTPException, File
from transformers import AutoTokenizer, AutoFeatureExtractor, AutomaticSpeechRecognitionPipeline, Wav2Vec2ForCTC
    
parser = ConfigArgParser()

parser.add_argument('--asr-model', default='jonatasgrosman/wav2vec2-xls-r-1b-english', type=str, help='path to model, service name, or json config file')
parser.add_argument('--wavs', default=None, type=str, nargs='+', help='path to input wav/ogg/flac file')
parser.add_argument('--output-dir')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)
    
# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()
    
model_name = args.asr_model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1
model = Wav2Vec2ForCTC.from_pretrained(model_name)
if device >= 0:
    model = model.to(f"cuda:{device}")
asr = AutomaticSpeechRecognitionPipeline(feature_extractor=feature_extractor, 
                                          model=model, 
                                          tokenizer=tokenizer, 
                                          device=device,
                                          chunk_length_s=5, 
                                          stride_length_s=(4, 2))

import os 

# run transcription

hotword_weight = 10

hotwords = ["zordon", "follow", "me", "often", "creator", "favorite", 
"is", "in", "hitchbot", "ever", "are", "allowed", "it's", 
"style", "what", "for", "movies", "away", "mark", "ar", "di" 
"python", "call", "music", "you", "has", "was", "do", "safe", "run", 
"get", "self", "driving", "cars", "why", "who", "robots", "zuckerberg", 
"language", "lunch", "be", "invented", "killed", "compiler", "kind", "ate", 
"your", "like", "salad", "created", "did", "of", "programming", "robot", 
"shouldn't", "person", "so", "the", "angry", "yes", "no", "up", "down", "left", "we", 
"on", "off", "to", "go"]

for audio_path in args.wavs:
    print(audio_path)
    if os.path.isfile(os.path.join(args.output_dir, os.path.basename(audio_path) + ".txt")): 
        continue
    try:
        st = time.time()
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0)
        result = asr(audio.numpy(), hotwords=hotwords, hotword_weight=hotword_weight)
        print(result["text"])        
        end = time.time()
        print('Took', str(end-st))
        # save TXT
        print('Saving' ,os.path.join(args.output_dir, os.path.basename(audio_path) + ".txt"))
        with open(os.path.join(args.output_dir, os.path.basename(audio_path) + ".txt"), "a", encoding="utf-8") as txt:
            print(result["text"], file=txt)
    except Exception as e:
        print('ERROR', audio_path, str(e))
        continue


