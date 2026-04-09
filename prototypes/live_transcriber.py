import queue
import threading
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ===== CONFIG =====
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "small"   # try "tiny" or "base" if it's too slow
DEVICE = "cpu"         # "cuda" if you have GPU
COMPUTE_TYPE = "int8"  # good for CPU
ROLLING_WINDOW_SECONDS = 10.0   # how much past audio to keep
UPDATE_INTERVAL_SECONDS = 1.5   # how often to re-transcribe
# ==================


def print_inline(text: str):
    """Print text on a single line that keeps updating."""
    sys.stdout.write("\r" + text + " " * 10)
    sys.stdout.flush()


class LiveTranscriber:
    def __init__(self):
        print("Loading model, this might take a bit...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.running = False

        # Rolling audio buffer (numpy array of float32)
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        # Store previous transcript to only show new part
        self.last_transcript = ""

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\n[Audio status] {status}", file=sys.stderr)

        # indata shape: (frames, channels)
        mono = indata[:, 0].copy()
        self.audio_queue.put(mono)

    def _audio_consumer(self):
        """Consumes raw audio from the queue and updates rolling buffer."""
        max_len = int(ROLLING_WINDOW_SECONDS * SAMPLE_RATE)

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Append new chunk to audio buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

            # Keep only the last max_len samples
            if len(self.audio_buffer) > max_len:
                self.audio_buffer = self.audio_buffer[-max_len:]

    def _transcription_loop(self):
        """Periodically transcribe the rolling buffer."""
        while self.running:
            if len(self.audio_buffer) == 0:
                time.sleep(UPDATE_INTERVAL_SECONDS)
                continue

            # Copy current buffer to avoid it changing mid-transcription
            buffer_copy = self.audio_buffer.copy()

            # Faster-Whisper accepts raw audio as a NumPy array at 16kHz
            segments, info = self.model.transcribe(
                buffer_copy,
                language="en",        # English only
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False
            )

            full_text = "".join(s.text for s in segments).strip()

            # Only show the newly-added part
            new_part = full_text[len(self.last_transcript):].strip()
            if new_part:
                self.last_transcript = full_text
                print_inline(full_text)

            time.sleep(UPDATE_INTERVAL_SECONDS)

    def start(self):
        self.running = True

        # Start consumer thread
        self.audio_thread = threading.Thread(target=self._audio_consumer, daemon=True)
        self.audio_thread.start()

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.transcription_thread.start()

        # Open microphone stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self.audio_callback,
        ):
            print("🎤 Live transcription started. Speak into your mic.")
            print("Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")
                self.running = False
                time.sleep(0.5)


if __name__ == "__main__":
    lt = LiveTranscriber()
    lt.start()
    print("\nFinal transcript:")
    print(lt.last_transcript)
