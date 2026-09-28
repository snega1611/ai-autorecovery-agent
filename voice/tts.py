from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path

TTS_ENABLED = os.environ.get("TTS_ENABLED", "true").lower() == "true"
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")            # path to the piper executable
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium.onnx")


def speak(text: str) -> None:
    """Synthesizes `text` and plays it. Falls back to printing if TTS is
    disabled or Piper isn't found, so the demo never hard-fails on audio
    setup issues."""
    if not TTS_ENABLED:
        print(f"[agent says]: {text}")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        proc = subprocess.run(
            [PIPER_BIN, "--model", PIPER_VOICE, "--output_file", wav_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="ignore"))

        _play_wav(wav_path)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[TTS unavailable ({e}), falling back to text]")
        print(f"[agent says]: {text}")
    finally:
        try:
            os.unlink(wav_path)
        except (OSError, NameError):
            pass


def _play_wav(path: str) -> None:
    import soundfile as sf
    import sounddevice as sd
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()


if __name__ == "__main__":
    speak("Hello, this is a test of the local text to speech system.")
