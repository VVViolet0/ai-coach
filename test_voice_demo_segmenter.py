from voice_demo.segmenter import FRAME_BYTES, VadSegmenter


class PatternVad:
    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.index = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        del frame, sample_rate
        if self.index >= len(self.pattern):
            return False
        value = self.pattern[self.index]
        self.index += 1
        return value


def frame() -> bytes:
    return b"\x00" * FRAME_BYTES


def loud_frame() -> bytes:
    return (1000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def test_segmenter_emits_start_and_end_after_silence():
    pattern = [False] * 10 + [True] * 30 + [False] * 40
    segmenter = VadSegmenter(vad=PatternVad(pattern))

    events = []
    for is_speech in pattern:
        events.extend(segmenter.process_frame(loud_frame() if is_speech else frame()))

    event_types = [event_type for event_type, _segment in events]
    assert event_types == ["vad_start", "vad_end"]

    segment = events[-1][1]
    assert segment is not None
    assert segment.duration_ms >= 300
    assert 0 <= segment.start_ms <= 200


def test_segmenter_drops_too_short_segments():
    pattern = [True] * 2 + [False] * 30
    segmenter = VadSegmenter(vad=PatternVad(pattern))

    events = []
    for is_speech in pattern:
        events.extend(segmenter.process_frame(loud_frame() if is_speech else frame()))

    assert events == []


def test_segmenter_keeps_short_confirmation_sized_segment_with_new_defaults():
    pattern = [True] * 10 + [False] * 40
    segmenter = VadSegmenter(vad=PatternVad(pattern))

    events = []
    for is_speech in pattern:
        events.extend(segmenter.process_frame(loud_frame() if is_speech else frame()))

    event_types = [event_type for event_type, _segment in events]
    assert event_types == ["vad_start", "vad_end"]
    assert events[-1][1] is not None


def test_segmenter_flushes_active_segment():
    pattern = [False] * 5 + [True] * 30
    segmenter = VadSegmenter(vad=PatternVad(pattern))

    for is_speech in pattern:
        segmenter.process_frame(loud_frame() if is_speech else frame())

    segment = segmenter.flush()
    assert segment is not None
    assert segment.duration_ms >= 300


def test_segmenter_ignores_low_energy_vad_positives():
    pattern = [True] * 50
    segmenter = VadSegmenter(vad=PatternVad(pattern), speech_energy_threshold=350.0)

    events = []
    for _ in pattern:
        events.extend(segmenter.process_frame(frame()))

    assert events == []
