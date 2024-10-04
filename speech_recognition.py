from flask import Flask
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import logging

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('start-listening')
def start_listening():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            while True:
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    emit('speech-to-text', text)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
    except sr.WaitTimeoutError:
        print("Timeout waiting for microphone")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    socketio.run(app)
