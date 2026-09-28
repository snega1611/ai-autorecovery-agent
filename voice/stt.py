from __future__ import annotations
import os
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base.en")
SAMPLE_RATE = 16000

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # int8 compute type keeps memory + CPU load low; downloads the
        # model to ~/.cache on first run
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def record_from_mic(seconds: float = 6.0) -> Path:
    """Records `seconds` of audio from the default mic and returns a path
    to a temp WAV file. Turn-based (fixed duration) rather than
    voice-activity-detected, to keep the demo dependency-light -- see the
    module docstring in conversation.py for why that's an acceptable
    simplification for a 10-hour build."""
    print(f"[listening for {seconds}s]...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return Path(tmp.name)


def transcribe(audio_path: Path) -> str:
    model = _get_model()
    segments, _info = model.transcribe(str(audio_path), language="en", vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def listen_and_transcribe(seconds: float = 6.0) -> str:
    path = record_from_mic(seconds)
    try:
        return transcribe(path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    text = listen_and_transcribe(5.0)
    print("Transcribed:", text)
