
from transformers import pipeline
import librosa
import soundfile as sf

v = 'sample.wav'
# v = 'Ray Dalio on US Dominance China Economy Inflation Future of Bridgewater.wav'

model = 'superb/wav2vec2-base-superb-er'
# model = 'hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD'

start_second = 0
end_second = 1   # 5
data, samplerate = librosa.load(v, offset=start_second, duration=(end_second - start_second))
sf.write('cut.wav', data, samplerate)
# au, samplerate = sf.read(v, dtype='float32', start=start_second, stop=end_second)

classifier = pipeline("audio-classification", model=model)
labels = classifier('cut.wav', top_k=5)
