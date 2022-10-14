#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import pyloudnorm as pyln
from multiprocessing import Process
import numpy as np

from jetson_voice import ASR, AudioInput, AudioWavStream, AudioSamplesStream, ConfigArgParser, list_audio_devices
from jetson_voice import TTS, ConfigArgParser, AudioOutput, list_audio_devices
from soundfile import SoundFile
import soundfile as sf
    
parser = ConfigArgParser()

parser.add_argument('--asr-model', default='quartznet', type=str, help='path to model, service name, or json config file')
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
    
# load the model

asr = ASR(args.asr_model)
if args.output_device:
    audio_device = AudioOutput(args.output_device, asr.sample_rate)
vad = ASR(args.vad_model)

vad(np.empty(4000))

# run transcription
background_detection_tresh = BACKGROUND_DETECTION_THRESH = 15
MAX_SAMPLES = 16_000*10

def play_audio(audio):
    audio_device.write(audio)

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
    while background_detection_tresh > 0:
        samples = next(stream)
        vad_results = vad(samples)
        print(vad_results, "| background_detection_tresh:", background_detection_tresh)
        if vad_results[0] == 'speech' and vad_results[1] > 0.9:
            background_detection_tresh = BACKGROUND_DETECTION_THRESH
        else:
            background_detection_tresh -= 1
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
    
    print('Audio shape:', asr_samples.shape)

    data, rate = sf.read(current_time+".wav") # load audio
    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(data, -1.0)

    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -24.0)

    output_wav_norm = SoundFile(current_time+".norm.wav", mode='w', samplerate=16_000, channels=1)
    output_wav_norm.write(loudness_normalized_audio)
    stream = AudioWavStream(current_time+".norm.wav",
                            sample_rate=asr.sample_rate, 
                            chunk_size=asr.chunk_size)

    # -------------------------------------------------------------------

    for samples in stream:
        end_transcription = False

        results = asr(samples)
        
        if asr.classification:
            print(f"class '{results[0]}' ({results[1]:.3f})")
        else:
            for transcript in results:
                print(transcript['text'])
                if transcript['end']:
                    print('-------------------------------------------------')
                    print("Final sentence:", transcript['text'])
                    # manda o comando pro nlu
                    
                    # # run the TTS
                    # if args.output_device and len(transcript['text']) > 5:
                    #     for run in range(args.warmup+1):
                    #         start = time.perf_counter()
                    #         audio = tts(transcript['text'])
                    #         stop = time.perf_counter()
                    #         latency = stop-start
                    #         duration = audio.shape[0]/tts.sample_rate
                    #         print(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
                            
                        # output the audio
                        # if args.output_device:
                            # audio_device.write(audio)

                    print('-------------------------------------------------')
                    end_transcription = True
        
        if end_transcription: break

    # ===================================================================

    r = input('Press q to exit or enter to run again: ')
    if r == 'q': 
        stop_transcription = True
        break
