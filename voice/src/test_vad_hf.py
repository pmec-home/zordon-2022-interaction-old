from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
from jetson_voice import ASR, AudioInput, AudioWavStream

stream = AudioWavStream(audio_path,
                        sample_rate=self.stt.sample_rate, 
                        chunk_size=self.stt.chunk_size)

for chunk in stream:
    output = pipeline(stream)
    for speech in output.get_timeline().support(): 
        print(speech)