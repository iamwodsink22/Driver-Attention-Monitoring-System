import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
volume = engine.getProperty('volume')
engine.setProperty('volume', 1.0)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
