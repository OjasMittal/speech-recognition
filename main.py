from speech_recognition import Recognizer, AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

recognizer = Recognizer()

with AudioFile('audio1.wav') as audio_file:
  audio = recognizer.record(audio_file)

text = recognizer.recognize_google(audio,language="hi-In")
print(text)
analyzer=SentimentIntensityAnalyzer()
if analyzer.polarity_scores(text)['compound']>0:
    print("Positive text")
else:
    print("Negative text")    