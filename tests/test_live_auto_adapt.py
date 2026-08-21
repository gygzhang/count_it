import numpy as np

import count_cv


class _FakeCapture:
    def __init__(self, frames):
        self._frames = iter(frames)

    def read(self):
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None


class _LiveSource:
    instances = []

    def __init__(self, source, fps=30.0):
        frames = [
            np.full((12, 16, 3), value, dtype=np.uint8)
            for value in range(6)
        ]
        self.cap = _FakeCapture(frames)
        self.live = True
        self.is_dir = False
        self.w, self.h, self.n, self.fps = 16, 12, 0, 300.0
        self.released = False
        self.instances.append(self)

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def release(self):
        self.released = True


def test_live_calibration_frames_are_replayed_once(monkeypatch):
    calibrated = []
    processed = []

    def fake_adapt(params, frames, method, w, h, ref=None, fps=30.0):
        calibrated.extend(int(frame[0, 0, 0]) for frame in frames)
        return params, {"calibration_frames": len(frames)}

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, frame):
            value = int(frame[0, 0, 0])
            processed.append(value)
            cx = 2.0 + value * 2.0
            return [(cx, 6.0, cx - 1, 5.0, 2.0, 2.0)]

    monkeypatch.setattr(count_cv, "FrameSource", _LiveSource)
    monkeypatch.setattr(count_cv, "auto_adapt_params", fake_adapt)
    monkeypatch.setattr(count_cv, "Detector", FakeDetector)

    result = count_cv.count_source(
        "0",
        {
            "method": "thresh",
            "auto_adapt": True,
            "calibration_frames": 3,
            "axis": "x",
            "flow": "pos",
            "line": 0.5,
            "min_hits": 1,
        },
    )

    assert result == 1
    assert calibrated == [0, 1, 2]
    assert processed == [0, 1, 2, 3, 4, 5]
    assert _LiveSource.instances[-1].released


def test_live_without_auto_adapt_has_no_startup_buffer(monkeypatch):
    processed = []

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, frame):
            processed.append(int(frame[0, 0, 0]))
            return []

    class FakeTracker:
        def __init__(self, *args, **kwargs):
            self.axis = "x"
            self.line_pos = 8
            self.count = 0
            self.tracks = []

        def update(self, dets, warming=False):
            pass

    monkeypatch.setattr(count_cv, "FrameSource", _LiveSource)
    monkeypatch.setattr(count_cv, "Detector", FakeDetector)
    monkeypatch.setattr(count_cv, "Tracker", FakeTracker)

    count_cv.count_source("0", {"method": "thresh", "auto_adapt": False})

    assert processed == [0, 1, 2, 3, 4, 5]
