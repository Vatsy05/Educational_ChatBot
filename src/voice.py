# src/voice.py

import speech_recognition as sr


def listen_voice():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.2

    try:
        microphone = sr.Microphone()
    except OSError:
        return "__NO_MIC__"

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(
            source,
            timeout=5,
            phrase_time_limit=10
        )

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "__UNRECOGNIZED__"
    except sr.RequestError:
        return "__API_ERROR__"
