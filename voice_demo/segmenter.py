from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, List, Optional, Protocol, Tuple


SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_SAMPLE = 2
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * BYTES_PER_SAMPLE


class VadLike(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        ...


@dataclass
class SpeechSegment:
    pcm: bytes
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class VadSegmenter:
    """Turns fixed-size PCM frames into speech segments."""

    def __init__(
        self,
        vad: VadLike,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        pre_roll_ms: int = 300,
        silence_ms: int = 800,
        max_segment_ms: int = 10_000,
        min_segment_ms: int = 180,
        start_trigger_ms: int = 100,
        speech_energy_threshold: float = 350.0,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * BYTES_PER_SAMPLE
        self.pre_roll_frames = max(0, pre_roll_ms // frame_ms)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_segment_frames = max(1, max_segment_ms // frame_ms)
        self.min_segment_frames = max(1, min_segment_ms // frame_ms)
        self.start_trigger_frames = max(1, start_trigger_ms // frame_ms)
        self.speech_energy_threshold = max(0.0, speech_energy_threshold)
        self.reset()

    def reset(self) -> None:
        self._frame_index = 0
        self._active = False
        self._start_frame = 0
        self._speech_frames = 0
        self._candidate_speech_frames = 0
        self._silent_frames = 0
        self._pre_roll: Deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._buffer: List[bytes] = []

    def process_frame(self, frame: bytes) -> List[Tuple[str, Optional[SpeechSegment]]]:
        if len(frame) != self.frame_bytes:
            raise ValueError(f"Expected {self.frame_bytes} bytes per frame, got {len(frame)}")

        events: List[Tuple[str, Optional[SpeechSegment]]] = []
        speech = self.vad.is_speech(frame, self.sample_rate) and self._frame_rms(frame) >= self.speech_energy_threshold

        if not self._active:
            self._pre_roll.append(frame)
            if speech:
                self._candidate_speech_frames += 1
            else:
                self._candidate_speech_frames = 0

            if self._candidate_speech_frames >= self.start_trigger_frames:
                self._active = True
                self._start_frame = max(0, self._frame_index - len(self._pre_roll) + 1)
                self._speech_frames = self._candidate_speech_frames
                self._silent_frames = 0
                self._buffer = list(self._pre_roll)
                events.append(("vad_start", None))
        else:
            self._buffer.append(frame)
            if speech:
                self._speech_frames += 1
                self._silent_frames = 0
            else:
                self._silent_frames += 1

            active_frames = self._frame_index - self._start_frame + 1
            if self._silent_frames >= self.silence_frames or active_frames >= self.max_segment_frames:
                segment = self._finalize_segment()
                events.append(("vad_end", segment))

        self._frame_index += 1
        return events

    def _frame_rms(self, frame: bytes) -> float:
        if not frame:
            return 0.0
        total = 0
        count = len(frame) // BYTES_PER_SAMPLE
        for index in range(0, len(frame), BYTES_PER_SAMPLE):
            sample = int.from_bytes(frame[index : index + BYTES_PER_SAMPLE], "little", signed=True)
            total += sample * sample
        return sqrt(total / max(1, count))

    def flush(self) -> Optional[SpeechSegment]:
        if not self._active:
            return None
        return self._finalize_segment()

    def _finalize_segment(self) -> Optional[SpeechSegment]:
        end_frame = self._frame_index + 1
        pcm = b"".join(self._buffer)
        segment = SpeechSegment(
            pcm=pcm,
            start_ms=self._start_frame * self.frame_ms,
            end_ms=end_frame * self.frame_ms,
        )

        self._active = False
        speech_frames = self._speech_frames
        self._speech_frames = 0
        self._candidate_speech_frames = 0
        self._silent_frames = 0
        self._buffer = []
        self._pre_roll.clear()

        if speech_frames < self.min_segment_frames:
            return None
        return segment
