from __future__ import annotations

import argparse
import math
import tempfile
import wave
from pathlib import Path

from faster_whisper import WhisperModel


def write_tone_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(samples):
            sample = int(0.12 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav.writeframesraw(sample.to_bytes(2, "little", signed=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test faster-whisper model loading and transcription.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    model_path = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    print(f"Loading model: {model_path}", flush=True)
    model = WhisperModel(
        model_path,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )
    print("Model loaded.", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        write_tone_wav(wav_path)
        print(f"Transcribing synthetic wav: {wav_path}", flush=True)
        segments, info = model.transcribe(str(wav_path), beam_size=1, vad_filter=False)
        text = "".join(segment.text for segment in segments).strip()
        print(f"Language: {getattr(info, 'language', None)}", flush=True)
        print(f"Text: {text!r}", flush=True)
        print("ASR smoke test completed.", flush=True)
    finally:
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
