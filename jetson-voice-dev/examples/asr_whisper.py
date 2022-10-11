#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from jetson_voice import ASR, AudioInput, ConfigArgParser, list_audio_devices
from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile
import soundfile as sf
import torch
import torchaudio
import whisper
from os import getenv
from fastapi import FastAPI, HTTPException, File
from transformers import AutoTokenizer, AutoFeatureExtractor, AutomaticSpeechRecognitionPipeline, Wav2Vec2ForCTC
    
parser = ConfigArgParser()

parser.add_argument('--tts-model', default='fastpitch_hifigan', type=str)
parser.add_argument('--vad-model') 
parser.add_argument('--wav', default=None, type=str, help='path to input wav/ogg/flac file')
parser.add_argument('--mic', default=None, type=str, help='device name or number of input microphone')
parser.add_argument("--output-device", default=None, type=str, help='output audio device to use')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')
parser.add_argument('--warmup', default=5, type=int, help='the number of warmup runs')

args = parser.parse_args()
print(args)
# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()

whisper_model = whisper.load_model("small")  # tiny small base medium large 
whisper_options = whisper.DecodingOptions(language="en", without_timestamps=True, beam_size=1)

vad = ASR(args.vad_model)

if args.output_device:
    audio_device = AudioOutput(args.output_device, 16_000)

# run transcription
background_detection_tresh = BACKGROUND_DETECTION_THRESH = 10
MAX_SAMPLES = 16_000*5

def play_audio(audio):
    try:
        if len(audio) > 16_000:
            audio_device.write(audio)
    except: pass

stop_transcription = False
while not stop_transcription:    
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)

    # ===================================================================
    # create the audio input stream
    stream = AudioInput(wav=args.wav, mic=args.mic, 
                        sample_rate=vad.sample_rate, 
                        chunk_size=vad.chunk_size)

    # -------------------------------------------------------------------

    asr_samples = []
    start = False
    while background_detection_tresh > 0:
        samples = next(stream)

        vad_results = vad(samples)

        print(vad_results, "| background_detection_tresh:", background_detection_tresh)
        if vad_results[0] == 'speech' and vad_results[1] > 0.9:
            start = True
            background_detection_tresh = BACKGROUND_DETECTION_THRESH
        else:
            background_detection_tresh -= 1
        if start:
            asr_samples = [*asr_samples, *samples]
        if len(asr_samples) > MAX_SAMPLES:
            print('Stop recording (max samples reached)')
            break
    
    background_detection_tresh = BACKGROUND_DETECTION_THRESH

    # ===================================================================

    stream.close()
    print('\naudio stream closed.')

    output_wav = SoundFile(current_time+".wav", mode='w', samplerate=16_000, channels=1)
    output_wav.write(asr_samples)
    asr_samples = np.array(asr_samples)

    if args.output_device:
        p = Process(target=play_audio, args=(asr_samples, ))
        p.start()
    
    if not len(asr_samples) > 16_000//2: continue
    print('Audio shape:', asr_samples.shape)

    data, rate = sf.read(current_time+".wav") # load audio
    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(data, -1.0)

    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -18.0)

    output_wav_norm = SoundFile(current_time+".norm.wav", mode='w', samplerate=16_000, channels=1)
    output_wav_norm.write(loudness_normalized_audio)
    
    print("Loading", current_time+".norm.wav")
    st = time.time()
    audio = whisper.load_audio(file=current_time+".norm.wav", sr=16_000)
    audio = whisper.pad_or_trim(audio.flatten())
    audio = torch.from_numpy(audio).to("cuda")
    
    mel = whisper.log_mel_spectrogram(audio)
    result = whisper_model.decode(mel, whisper_options)
    end = time.time()

    print("Whisper transcription:", result.text, f" | took {end-st} seconds")

    # ===================================================================

    r = input('Press q to exit or enter to run again: ')
    if r == 'q': 
        stop_transcription = True
        break
