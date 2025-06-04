from TTS.api import TTS
import sounddevice as sd
import os
import sys
import warnings

# Create a directory for models if it doesn't exist
os.makedirs("tts_models", exist_ok=True)

# Initialize TTS with VCTK model
print("Initializing TTS...")
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)

# Define selected speakers
speakers = [
    "p260",  # Voice 3
    "p270",  # Voice 5
    "p306",  # Voice 9
    "p310"   # Voice 10
]

print(f"\nLoaded {len(speakers)} selected voices")
print("Type 'exit' to quit the program")

try:
    while True:
        # Get text input
        text = input("\nEnter text to synthesize (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            print("\nExiting program...")
            break
        if not text:
            continue
            
        # Synthesize with each speaker
        for i, speaker_id in enumerate(speakers, 1):
            print(f"\nVoice {i}/{len(speakers)} - Speaker {speaker_id}")
            wav = tts.tts(text, speaker=speaker_id)
            print("Playing audio...")
            sd.play(wav, samplerate=22050)
            sd.wait()
            print("Done.")
            
except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1) 