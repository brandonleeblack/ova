from ollama import chat
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os
import queue
import sys
import json
import numpy as np
from threading import Lock, Thread, Event
from TTS.api import TTS

class AudioProcessor:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.stop_event = Event()
        self.playback_thread = None
        self.speech_complete = Event()
        self.speech_complete.set()
        self.is_playing = False

    def start_playback_thread(self):
        self.playback_thread = Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is not None and not self.is_playing:
                    self.is_playing = True
                    self.speech_complete.clear()
                    
                    # Play audio and wait for completion
                    sd.play(audio_data, samplerate=22050)
                    sd.wait()
                    
                    self.is_playing = False
                    if self.audio_queue.empty():
                        self.speech_complete.set()
            except queue.Empty:
                continue
            except Exception:
                self.is_playing = False
                self.speech_complete.set()

    def wait_for_speech_complete(self, timeout=None):
        return self.speech_complete.wait(timeout)

    def queue_audio(self, audio_data):
        if not self.is_playing:
            self.audio_queue.put(audio_data)

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def cleanup(self):
        self.stop_event.set()
        self.clear_queue()
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/glow-tts", gpu=False):
        os.makedirs("tts_models", exist_ok=True)
        TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        self.tts.tts("Hello")  # Pre-warm

    def clean_text(self, text):
        text = text.replace("...", ".").replace("*", " ").replace('"', '"')
        text = ''.join(char for char in text if (char.isprintable() and ord(char) < 128) or char == "'")
        return ' '.join(text.split())

    def get_sentences(self, text):
        sentences = []
        sentence = ""
        for char in text:
            sentence += char
            if char in [".", "!", "?"]:
                if len(sentence.strip()) > 1:
                    sentences.append(sentence.strip())
                sentence = ""
        if sentence.strip():
            sentences.append(sentence.strip())
        return sentences

    def process_text(self, text, audio_processor):
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return
                
            wav = self.tts.tts(cleaned_text)
            if isinstance(wav, np.ndarray):
                if wav.dtype != np.float32:
                    wav = wav.astype(np.float32)
            else:
                wav = np.array(wav, dtype=np.float32)
                
            audio_processor.queue_audio(wav)
        except Exception:
            pass

    def process_stream(self, response, audio_processor):
        message_so_far = ""
        final_message = ""
        sentences_to_process = []
        processed_text = set()
        
        try:
            for chunk in response:
                text = chunk['message']['content']
                message_so_far += text
                final_message += text

                sentences = self.get_sentences(message_so_far)
                
                if len(sentences) > 0:
                    for sentence in sentences[:-1]:
                        if sentence in processed_text:
                            continue
                            
                        message_so_far = message_so_far.replace(sentence, "")
                        sentences_to_process.append(sentence)
                        processed_text.add(sentence)
                        
                        if len(sentences_to_process) >= 2:
                            for sentence in sentences_to_process:
                                self.process_text(sentence, audio_processor)
                            sentences_to_process = []
            
            for sentence in sentences_to_process:
                self.process_text(sentence, audio_processor)
            
            if message_so_far and message_so_far not in processed_text:
                self.process_text(message_so_far, audio_processor)
            
            return final_message
        except Exception:
            return final_message

class VoiceAssistant:
    def __init__(self, model_name="tts_models/en/ljspeech/glow-tts", gpu=False):
        self.audio_processor = AudioProcessor()
        self.tts = TextToSpeech(model_name, gpu)
        self.audio_processor.start_playback_thread()
        self.listen = False
        self.prompt = ""
        self.commands = {
            "listen": self.on_listen_detected,
            "quiet": self.on_quiet_detected,
            "prompt": self.on_prompt_detected,
            "reset": self.on_reset_detected,
            "terminate": self.on_terminate_detected,
        }

    def on_listen_detected(self, text):
        self.listen = True

    def on_quiet_detected(self, text):
        self.listen = False

    def on_prompt_detected(self, text):
        self.prompt += text.replace("prompt", "")

    def on_reset_detected(self, text):
        self.prompt = ""
        self.tts.process_text("Resetting prompt", self.audio_processor)

    def on_terminate_detected(self, text):
        sys.exit(0)

    def process_command(self, text):
        for keyword, action in self.commands.items():
            if keyword in text.lower():
                action(text)

    def listen_for_command(self):
        try:
            model = Model(lang="en-us")
            device_info = sd.query_devices(device=0, kind="input")
            samplerate = int(device_info["default_samplerate"])
            blocksize = samplerate // 2
            
            with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, 
                                 device=0, dtype="int16", channels=1, 
                                 callback=self.audio_callback):
                rec = KaldiRecognizer(model, samplerate)
                print("Listening...")
                
                while True:
                    try:
                        data = self.audio_processor.audio_queue.get(timeout=1.0)
                        if rec.AcceptWaveform(data):
                            text = json.loads(rec.Result())['text']
                            if len(text) > 0:
                                print(f"You said: {text}")
                                self.process_command(text)
                                if self.listen:
                                    response = self.get_ollama_response(text)
                                    self.tts.process_stream(response, self.audio_processor)
                                    self.audio_processor.wait_for_speech_complete()
                                    time.sleep(1)
                                    self.audio_processor.clear_queue()
                                    print("Listening...")
                        else:
                            text = json.loads(rec.PartialResult())['partial']
                            if len(text) > 0:
                                print(f"Partial: {text}")
                    except queue.Empty:
                        continue
                    except Exception:
                        continue
        except Exception:
            sys.exit(1)

    def audio_callback(self, indata, frames, time, status):
        if status and status.input_overflow:
            self.audio_processor.clear_queue()
        self.audio_processor.audio_queue.put(bytes(indata))

    def get_ollama_response(self, text):
        try:
            return chat(
                model="gemma3:1b",
                messages=[{'role': 'user', 'content': text + self.prompt}],
                stream=True,
            )
        except Exception:
            return None

    def cleanup(self):
        self.audio_processor.cleanup()

def main():
    print("Available voice commands:")
    print("- 'listen': Start listening for commands")
    print("- 'quiet': Stop listening")
    print("- 'prompt': Add to the system prompt")
    print("- 'reset': Reset the system prompt")
    print("- 'terminate': Exit the program")

    assistant = VoiceAssistant()
    try:
        assistant.listen_for_command()
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 