import cv2
import numpy as np

from count_cv import DEFAULT_PARAMS, auto_adapt_params


def moving_rect_frames(size=(240, 160), rect=(24, 18), step=(12, 0), count=12):
    w, h = size
    rw, rh = rect
    frames = []
    for i in range(count):
        frame = np.full((h, w, 3), 128, np.uint8)
        x, y = 8 + step[0] * i, 60 + step[1] * i
        cv2.rectangle(frame, (x, y), (x + rw - 1, y + rh - 1), (0, 0, 255), -1)
        frames.append(frame)
    return frames


def test_auto_adapt_estimates_object_scale_and_motion():
    frames = moving_rect_frames()
    params = {
        **DEFAULT_PARAMS, "method": "color", "sat_thresh": 20,
        "axis": "x", "scale": 1.0,
    }
    adapted, diag = auto_adapt_params(
        params, frames, "color", 240, 160, fps=120)

    assert 300 <= diag["typical_box_area_px"] <= 500
    assert abs(diag["motion_dx_px_per_frame"] - 12) <= 1
    assert abs(diag["motion_dy_px_per_frame"]) <= 1
    assert 20 <= adapted["max_dist"] <= 30
    assert adapted["min_area"] < 100
    assert adapted["morph_kernel"] in (1, 3)
    assert adapted["min_hits"] >= 2


def test_auto_adapt_changes_with_size_and_speed():
    base = {**DEFAULT_PARAMS, "method": "color", "sat_thresh": 20}
    small, _ = auto_adapt_params(
        base, moving_rect_frames(rect=(12, 10), step=(4, 0)),
        "color", 240, 160, fps=60)
    large, _ = auto_adapt_params(
        base, moving_rect_frames(rect=(45, 35), step=(20, 0), count=8),
        "color", 240, 160, fps=300)

    assert large["min_area"] > small["min_area"]
    assert large["morph_kernel"] >= small["morph_kernel"]
    assert large["max_dist"] > small["max_dist"]


def test_auto_adapt_keeps_parameters_without_foreground():
    frames = [np.full((80, 100, 3), 128, np.uint8) for _ in range(5)]
    params = {**DEFAULT_PARAMS, "method": "color", "sat_thresh": 20}
    adapted, diag = auto_adapt_params(params, frames, "color", 100, 80)

    assert adapted == params
    assert "status" in diag
