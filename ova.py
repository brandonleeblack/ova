from ollama import chat
import unicodedata
import time
import pyaudio
import json
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os
import argparse
import logging
from TTS.api import TTS
import numpy as np
from threading import Lock, Thread, Event

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.stop_event = Event()
        self.playback_thread = None
        self.speech_complete = Event()
        self.speech_complete.set()

    def start_playback_thread(self):
        """Start the audio playback thread"""
        self.playback_thread = Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        """Worker thread for playing audio"""
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is not None:
                    self.speech_complete.clear()
                    # Convert audio data to float32 numpy array if it's not already
                    if isinstance(audio_data, bytes):
                        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif isinstance(audio_data, np.ndarray):
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                    else:
                        logger.error(f"Unsupported audio data type: {type(audio_data)}")
                        continue
                    
                    sd.play(audio_data, samplerate=22050)
                    sd.wait()
                    if self.audio_queue.empty():
                        self.speech_complete.set()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in playback thread: {str(e)}")
                self.speech_complete.set()

    def wait_for_speech_complete(self, timeout=None):
        """Wait for current speech to complete"""
        return self.speech_complete.wait(timeout)

    def queue_audio(self, audio_data):
        """Queue audio data for playback"""
        self.audio_queue.put(audio_data)

    def cleanup(self):
        """Cleanup resources"""
        self.stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/glow-tts", gpu=False):
        """Initialize the TTS engine"""
        try:
            os.makedirs("tts_models", exist_ok=True)
            # Force download the model first
            TTS(model_name=model_name, progress_bar=True, gpu=gpu)
            # Then initialize
            self.tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
            logger.info("TTS engine initialized successfully")
            
            # Pre-warm the model
            logger.info("Pre-warming TTS model...")
            self.tts.tts("Hello")
            logger.info("TTS model pre-warmed")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean text for TTS processing"""
        # Remove special characters and normalize, but preserve apostrophes
        text = text.replace("...", ".").replace("*", " ").replace('"', '"')
        # Remove any remaining special characters except apostrophes
        text = ''.join(char for char in text if (char.isprintable() and ord(char) < 128) or char == "'")
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def get_sentences(self, text):
        """Split text into sentences"""
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
        """Process text and convert to speech"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.warning("Empty text after cleaning, skipping TTS")
                return
                
            wav = self.tts.tts(cleaned_text)
            # Ensure wav is in the correct format
            if isinstance(wav, np.ndarray):
                if wav.dtype != np.float32:
                    wav = wav.astype(np.float32)
            else:
                wav = np.array(wav, dtype=np.float32)
                
            audio_processor.queue_audio(wav)
            logger.info(f"Audio queued for: {cleaned_text}")
        except Exception as e:
            logger.error(f"Error in process_text: {str(e)}")

    def process_stream(self, response, audio_processor):
        """Process streaming response and convert to speech"""
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
            
            # Process remaining sentences
            for sentence in sentences_to_process:
                self.process_text(sentence, audio_processor)
            
            # Process any remaining text
            if message_so_far and message_so_far not in processed_text:
                self.process_text(message_so_far, audio_processor)
            
            return final_message
        except Exception as e:
            logger.error(f"Error in process_stream: {str(e)}")
            return final_message

class VoiceAssistant:
    def __init__(self, model_name="tts_models/en/ljspeech/glow-tts", gpu=False):
        self.audio_processor = AudioProcessor()
        self.tts = TextToSpeech(model_name, gpu)
        self.audio_processor.start_playback_thread()
        self.listen = False
        self.prompt = ". Keep it short and conversational. "
        self.commands = {
            "listen": self.on_listen_detected,
            "quiet": self.on_quiet_detected,
            "prompt": self.on_prompt_detected,
            "reset": self.on_reset_detected,
            "terminate": self.on_terminate_detected,
        }

    def on_listen_detected(self, text):
        logger.info(f"[ACTION] Detected 'listen' in: {text}")
        self.listen = True

    def on_quiet_detected(self, text):
        logger.info(f"[ACTION] Detected 'quiet' in: {text}")
        self.listen = False

    def on_prompt_detected(self, text):
        logger.info(f"[ACTION] Detected 'prompt' in: {text}")
        self.prompt += text.replace("prompt", "")

    def on_reset_detected(self, text):
        logger.info(f"[ACTION] Detected 'reset' in: {text}")
        self.prompt = ". Keep it short and conversational. "
        self.tts.process_text("Resetting prompt", self.audio_processor)

    def on_terminate_detected(self, text):
        logger.info(f"[ACTION] Detected 'terminate' in: {text}")
        sys.exit(0)

    def process_command(self, text):
        """Process voice commands"""
        for keyword, action in self.commands.items():
            if keyword in text.lower():
                action(text)

    def listen_for_command(self):
        """Main listening loop"""
        try:
            model = Model(lang="en-us")
            device_info = sd.query_devices(device=0, kind="input")
            samplerate = int(device_info["default_samplerate"])
            blocksize = samplerate // 2
            
            with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, 
                                 device=0, dtype="int16", channels=1, 
                                 callback=self.audio_callback):
                rec = KaldiRecognizer(model, samplerate)
                logger.info("Listening...")
                
                while True:
                    try:
                        data = self.audio_processor.audio_queue.get(timeout=1.0)
                        if rec.AcceptWaveform(data):
                            text = json.loads(rec.Result())['text']
                            if len(text) > 0:
                                logger.info(f"You said: {text}")
                                self.process_command(text)
                                if self.listen:
                                    response = self.get_ollama_response(text)
                                    self.tts.process_stream(response, self.audio_processor)
                                    self.audio_processor.wait_for_speech_complete()
                                    time.sleep(1)
                                    self.audio_processor.audio_queue.queue.clear()
                                    logger.info("Listening...")
                        else:
                            text = json.loads(rec.PartialResult())['partial']
                            if len(text) > 0:
                                logger.info(f"Partial: {text}")
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error in listen_for_command: {str(e)}")
            sys.exit(1)

    def audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice"""
        if status:
            if status.input_overflow:
                logger.warning("Input buffer overflow - clearing queue")
                self.audio_processor.audio_queue.queue.clear()
            else:
                logger.error(status)
        self.audio_processor.audio_queue.put(bytes(indata))

    def get_ollama_response(self, text):
        """Get response from Ollama"""
        logger.info(f"Sent: {text}")
        try:
            return chat(
                model="gemma3:1b",
                messages=[{'role': 'user', 'content': text + self.prompt}],
                stream=True,
            )
        except Exception as e:
            logger.error(f"Error in get_ollama_response: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        self.audio_processor.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant with TTS Options')
    parser.add_argument('--model', default='tts_models/en/ljspeech/glow-tts',
                        help='TTS model to use (default: ljspeech/glow-tts)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')
    args = parser.parse_args()

    logger.info("Available voice commands:")
    logger.info("- 'listen': Start listening for commands")
    logger.info("- 'quiet': Stop listening")
    logger.info("- 'prompt': Add to the system prompt")
    logger.info("- 'reset': Reset the system prompt")
    logger.info("- 'terminate': Exit the program")

    assistant = VoiceAssistant(model_name=args.model, gpu=args.gpu)
    try:
        assistant.listen_for_command()
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main() 