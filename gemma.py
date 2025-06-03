import pyttsx3
from ollama import chat
import unicodedata
import time
import pyaudio
import json
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"  # Change to your preferred model if needed
prompt = ". Keep it short and conversational. "
listen = False
q = queue.Queue()

def say(sentence):
    engine = pyttsx3.init()
    #engine.setProperty('rate', 120)
    #engine.setProperty('pitch', 50)
    engine.setProperty('voice', 'en+f2')
    engine.say(sentence)
    engine.runAndWait()  # First call may initialize audio system
    engine.stop()        # Properly stop the engine
    del engine           # Release resources
    print("Audio generated for: ", sentence)
    time.sleep(0.5)

def say_stream(response):
    message_so_far = ""
    final_message = ""
    for chunk in response:
        text = clean_text(chunk['message']['content'])
        message_so_far += text
        final_message += text

        # Get sentences
        sentences = get_sentences(message_so_far)

        if len(sentences) > 0:
            for sentence in sentences[:-1]:
                message_so_far = message_so_far.replace(sentence, "")
                say(sentence)
    say(message_so_far)
    return final_message

def get_sentences(input : str):
    sentences = []
    sentence = ""
    for char in input:
        sentence += char
        if char in [".", "!", "?"]:
            if len(sentence) > 1:
                sentences.append(sentence)
            else:
                sentences[-1] += sentence
            sentence = ""
    return sentences

def on_listen_detected(text):
    print(f"[ACTION] Detected 'listen' in: {text}")
    global listen
    listen = True

def on_quiet_detected(text):
    print(f"[ACTION] Detected 'quiet' in: {text}")
    global listen
    listen = False

def on_prompt_detected(text):
    print(f"[ACTION] Detected 'prompt' in: {text}")
    global prompt
    prompt += text.replace("prompt", "")

def on_reset_detected(text):
    print(f"[ACTION] Detected 'reset' in: {text}")
    global prompt
    prompt = " Keep it short and conversational. "
    say("Resetting prompt")

def on_terminate_detected(text):
    print(f"[ACTION] Detected 'terminate' in: {text}")
    exit(0)

KEYWORDS = {
    "listen": on_listen_detected,
    "quiet": on_quiet_detected,
    "prompt": on_prompt_detected,
    "reset": on_reset_detected,
    "terminate": on_terminate_detected,
    # Add more keywords and functions as needed
}

def listen_for_keywords(text):
    for keyword, action in KEYWORDS.items():
        if keyword in text.lower():
            action(text)

def clean_text(text):
    # remove bad characters
    text = text.replace("...",".").replace("*", " ").replace('"', '"').replace("'", "'")
    # Normalize to NFKD and encode to ASCII, ignoring errors (removes emojis and other non-ASCII)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen_for_command():
    model = Model(lang="en-us")
    device_info = sd.query_devices(device=0, kind="input")
    # soundfile expects an int, sounddevice provides a float:
    samplerate = int(device_info["default_samplerate"])
    print(samplerate)
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=0, 
                           dtype="int16", channels=1, callback=callback):
        rec = KaldiRecognizer(model, samplerate)
        print("Listening...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):#accept waveform of input voice
                # Parse the JSON result and get the recognized text
                text = json.loads(rec.Result())['text']
                if len(text) > 0:
                    print("You said: " + text)
                    listen_for_keywords(text)
                    if listen:
                        response = ollama_command(text + prompt)
                        say_stream(response)
                        q.queue.clear()
                        time.sleep(1)
                        print("Listening...")
            else:
                text = json.loads(rec.PartialResult())['partial']
                if len(text) > 0:
                    print("Partial: " + text)

def ollama_command(text):
    print("Sent: "+ text)
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': text}],
        stream=True,
    )
    return response

def main():
    listen_for_command()  

if __name__ == "__main__":
    main()
