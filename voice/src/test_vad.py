import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts, _, read_audio,_, _) = utils


wav = read_audio("/home/athome/zordon-2022/zordon-2022-interaction/voice/src/23_17_17.wav")
# full audio
# get speech timestamps from full audio file
speech_timestamps = get_speech_ts(wav, model)
pprint(speech_timestamps)