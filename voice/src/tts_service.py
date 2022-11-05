#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import readline

from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile

import uvicorn
from fastapi import FastAPI, HTTPException, File
    
from pydantic import BaseModel

class Msg(BaseModel):
    msg : str

# load the model
tts = TTS("/home/athome/zordon-2022/zordon-2022-interaction/voice/src/data/networks/tts/fastpitch_hifigan/fastpitch_hifigan.json")

audio_device = AudioOutput(24, tts.sample_rate)
    
# run the TTS 
for run in range(5):
    start = time.perf_counter()
    audio = tts('text')
    stop = time.perf_counter()
    latency = stop-start
    duration = audio.shape[0]/tts.sample_rate
    print(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
    
app = FastAPI()

@app.get('/health')
async def health_check():
    return 'OK'

@app.post('/speak')
async def speak(msg:Msg):
    # output the audio
    print(f'Speaking \"{msg.msg}\"')
    audio = tts(msg.msg)
    audio_device.write(audio)

@app.get('/hi')
async def hi():
    # output the audio
    audio = tts("Hello")
    audio_device.write(audio)
    return 'Hello'

# @app.post('/save')
# async def save(msg:str, wav_path:str):
#     wav_path = os.path.join(args.output_wav, f'{wav_count}.wav') if wav_is_dir else args.output_wav
#     wav = SoundFile(wav_path, mode='w', samplerate=tts.sample_rate, channels=1)
#     wav.write(audio)
#     wav.close()
#     wav_count += 1
#     print(f"\nWrote audio to {wav_path}")

# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000)