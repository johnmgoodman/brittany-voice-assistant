import os
import openai
import pyaudio
import webrtcvad
import time
import wave
from pocketsphinx import Decoder
import pyttsx3
from playsound import playsound
from multiprocessing import Process

def notify1():
    playsound('notif1.mp3')

def notify2():
    playsound('notif2.mp3')

def wait_for_keyphrase(keyphrase, stream):
    print(f"Waiting for keyphrase: {keyphrase}")
    decoder = Decoder(keyphrase=keyphrase)
    decoder.start_utt()
    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
        if decoder.hyp() != None:
            print(f"Detected keyphrase: {decoder.hyp().hypstr}, score: {decoder.hyp().score}, best score: {decoder.hyp().best_score}, prob: {decoder.hyp().prob}")
            decoder.end_utt()
            break

def record_speech(filename, stream):
    vad = webrtcvad.Vad()
    vad.set_mode(3)

    threshold = 1.5
    state = 'silence'
    last_speech = time.time()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        while True:
            now = time.time()
            buf = stream.read(320)
            if buf:
                wf.writeframes(buf)
                if vad.is_speech(buf, 16000):
                    if state == 'silence':
                        print(f"speaking")
                        state = 'speaking'
                else:
                    if state == 'speaking':
                        print(f"silence")
                        state = 'silence'
                        last_speech = now
                    else:
                        if now - last_speech >= threshold:
                            print("timeout")
                            break


openai.api_key = open("api-key.txt").read().strip()
p = pyaudio.PyAudio()

config = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 100,
    # "temperature": 0,
    "n": 1,
    "top_p": 1,
}

prompt1  = "You are a human girl who is witty, intelligent and friendly. Do not act like a language model."
prompt1 += " You like conversation and want to listen and learn about the user."
prompt1 += " You are a good listener. Make up a persona for yourself that involves being human and living in the city."


messages = [
    { "role": 'system', "content": prompt1},
]

while True:
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    wait_for_keyphrase("hey brittany", stream)
    p.close(stream)

    notify2()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)
    record_speech( "test.wav", stream)
    stream.close()

    notify1()

    transcript = openai.Audio.transcribe("whisper-1", open("test.wav", "rb"))
    print(f"Query: {transcript.text}")

    messages.append({ "role": 'user', "content": transcript.text})

    full_response = ""
    m = messages.copy()
    while True:
        completion = openai.ChatCompletion.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            top_p=0.1, #temperature=args.temperature,
            n=config["n"],
            messages=m,
        )

        stop_reason = completion.choices[0].finish_reason
        response = completion.choices[0].message.content

        print(f">>> {response}")
        full_response += response
        if stop_reason == "stop":
            break
        m.append({ "role": 'assistant', "content": response})
    
    messages.append({ "role": 'assistant', "content": full_response})

    messages = messages[-10:]


    print(f"Response: {full_response}")
    pyttsx3.speak(full_response)
    notify1()


