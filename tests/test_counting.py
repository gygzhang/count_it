import cv2
import numpy as np
import pytest

from counting import count_source, load_or_make_ref


def test_count_source_validates_roi_against_processed_dimensions(tmp_path):
    assert cv2.imwrite(str(tmp_path / "frame1.jpg"), np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="roi"):
        count_source(str(tmp_path), {"roi": "20,0,30,5"})


def test_count_source_rejects_unopened_writer_before_processing(tmp_path, monkeypatch):
    assert cv2.imwrite(str(tmp_path / "frame1.jpg"), np.zeros((10, 10, 3), np.uint8))
    destination = tmp_path / "output.mp4"
    processed = []

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            processed.append("released")

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: ClosedWriter())

    with pytest.raises(RuntimeError, match=str(destination)):
        count_source(str(tmp_path), {"method": "color"}, save=str(destination))
    assert processed == ["released"]


def test_load_or_make_ref_rejects_existing_unreadable_bg_ref(tmp_path):
    ref_path = tmp_path / "bad-reference.jpg"
    ref_path.write_bytes(b"not an image")

    with pytest.raises(RuntimeError) as exc_info:
        load_or_make_ref(
            {"bg_ref": str(ref_path)},
            [np.zeros((10, 10, 3), dtype=np.uint8)], 10, 10)

    assert str(ref_path) in str(exc_info.value)
