import cv2
import numpy as np
import pytest

from counting import count_source
from sources import FrameSource, natural_key


def write_image(path, shape=(8, 12, 3)):
    assert cv2.imwrite(str(path), np.zeros(shape, dtype=np.uint8))


def test_natural_key_orders_numeric_names():
    names = ["frame10.jpg", "frame2.jpg", "frame1.jpg"]
    assert sorted(names, key=natural_key) == ["frame1.jpg", "frame2.jpg", "frame10.jpg"]


def test_directory_frame_rejects_dimension_mismatch(tmp_path):
    write_image(tmp_path / "frame1.jpg", (8, 12, 3))
    write_image(tmp_path / "frame2.jpg", (9, 12, 3))
    source = FrameSource(str(tmp_path))
    frames = source.frames()
    next(frames)
    with pytest.raises(RuntimeError, match=r"frame2\.jpg.*12x8.*12x9"):
        next(frames)


def test_directory_frame_rejects_unreadable_image_when_read(tmp_path):
    write_image(tmp_path / "frame1.jpg")
    (tmp_path / "frame2.jpg").write_bytes(b"not an image")
    source = FrameSource(str(tmp_path))
    frames = source.frames()
    next(frames)
    with pytest.raises(RuntimeError, match="frame2.jpg"):
        next(frames)


def test_sample_rejects_dimension_mismatch(tmp_path):
    write_image(tmp_path / "frame1.jpg", (8, 12, 3))
    write_image(tmp_path / "frame2.jpg", (9, 12, 3))
    source = FrameSource(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"frame2\.jpg.*12x8.*12x9"):
        source.sample(1)


def test_sample_rejects_unreadable_image(tmp_path):
    write_image(tmp_path / "frame1.jpg")
    (tmp_path / "frame2.jpg").write_bytes(b"not an image")
    source = FrameSource(str(tmp_path))
    with pytest.raises(RuntimeError, match="frame2.jpg"):
        source.sample(1)


def test_count_source_releases_outputs_on_late_read_error(tmp_path, monkeypatch):
    write_image(tmp_path / "frame1.jpg")
    (tmp_path / "frame2.jpg").write_bytes(b"not an image")

    released = {"source": False}

    class TrackingSource(FrameSource):
        def release(self):
            released["source"] = True
            super().release()

    class FakeWriter:
        def __init__(self):
            self.released = False
        def isOpened(self):
            return True


        def write(self, frame):
            pass

        def release(self):
            self.released = True

    writer = FakeWriter()
    destroyed = []
    monkeypatch.setattr("counting.FrameSource", TrackingSource)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))

    with pytest.raises(RuntimeError, match="frame2.jpg"):
        count_source(
            str(tmp_path), params={"method": "color"},
            save=str(tmp_path / "output.mp4"), show=True)

    assert released["source"]
    assert writer.released
    assert destroyed == [True]


def test_video_source_rejects_failed_read_before_reported_count(tmp_path, monkeypatch):
    path = tmp_path / "broken.mp4"

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 12,
                cv2.CAP_PROP_FRAME_HEIGHT: 8,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_FRAME_COUNT: 3,
            }
            return values.get(property_id, 0)

        def set(self, property_id, value):
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.index = int(value)

        def read(self):
            if self.index == 0:
                self.index += 1
                return True, np.zeros((8, 12, 3), dtype=np.uint8)
            self.index += 1
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    source = FrameSource(str(path))
    frames = source.frames()
    assert next(frames).shape == (8, 12, 3)
    with pytest.raises(RuntimeError, match=r"broken\.mp4"):
        next(frames)

    source = FrameSource(str(path))
    with pytest.raises(RuntimeError, match=r"broken\.mp4"):
        source.sample(1)
