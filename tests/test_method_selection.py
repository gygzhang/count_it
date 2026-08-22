import cv2
import numpy as np

from count_cv import (DEFAULT_PARAMS, _gray_background_band,
                      choose_method_frames, prepare_method)


def uniform_gray_frames(bg=120, obj=50, size=(160, 240), n=8):
    """Uniform gray belt with a single moving brightness-distinct object."""
    h, w = size
    frames = []
    for i in range(n):
        f = np.full((h, w, 3), bg, np.uint8)
        cv2.circle(f, (20 + i * 12, h // 2), 14, (obj, obj, obj), -1)
        frames.append(f)
    return frames


def saturated_frames(size=(160, 240), n=8):
    """Gray belt with a highly-saturated (red) object -> color method."""
    h, w = size
    frames = []
    for i in range(n):
        f = np.full((h, w, 3), 110, np.uint8)
        cv2.circle(f, (20 + i * 12, h // 2), 16, (0, 0, 255), -1)
        frames.append(f)
    return frames


def textured_gray_frames(size=(160, 240), n=8):
    """Wide-spread gray intensities (patterned belt) -> no tight band."""
    rng = np.random.default_rng(0)
    return [np.repeat(rng.integers(40, 210, size, dtype=np.uint8)[:, :, None],
                      3, axis=2) for _ in range(n)]


def test_auto_selects_thresh_for_uniform_gray_dark_object():
    method, info = choose_method_frames(uniform_gray_frames(bg=120, obj=50))
    assert method == "thresh"
    lo, hi = info["thresh_lo"], info["thresh_hi"]
    assert lo < 120 < hi          # uniform background sits inside the band
    assert 50 < lo                # dark object is below the band -> foreground
    assert 0 <= lo < hi <= 255


def test_auto_selects_thresh_for_bright_object():
    method, info = choose_method_frames(uniform_gray_frames(bg=120, obj=200))
    assert method == "thresh"
    assert info["thresh_hi"] < 200   # bright object is above the band


def test_auto_selects_color_when_saturated():
    method, info = choose_method_frames(saturated_frames())
    assert method == "color"
    assert "thresh_lo" not in info


def test_auto_keeps_bgsub_for_textured_background():
    method, _ = choose_method_frames(textured_gray_frames())
    assert method == "bgsub"
    assert _gray_background_band(textured_gray_frames()) is None


def test_prepare_method_writes_auto_thresh_band():
    P = {**DEFAULT_PARAMS, "method": "auto"}
    method, ref = prepare_method(P, uniform_gray_frames(bg=120, obj=50), 240, 160)
    assert method == "thresh"
    assert ref is None
    # Band is written back into the resolved params (default 50/205 replaced).
    assert P["thresh_lo"] < 120 < P["thresh_hi"]
    assert P["thresh_lo"] != DEFAULT_PARAMS["thresh_lo"]


def test_prepare_method_color_leaves_thresh_defaults():
    P = {**DEFAULT_PARAMS, "method": "auto"}
    method, _ = prepare_method(P, saturated_frames(), 240, 160)
    assert method == "color"
    assert P["thresh_lo"] == DEFAULT_PARAMS["thresh_lo"]
    assert P["thresh_hi"] == DEFAULT_PARAMS["thresh_hi"]


def _dark_object_folder(tmp_path, n=14, bg=190, obj=40, w=200, h=120):
    """Bright belt with one dark object crossing the centre line (Otsu case)."""
    d = tmp_path / "otsu"
    d.mkdir()
    for i in range(n):
        f = np.full((h, w, 3), bg, np.uint8)
        cv2.circle(f, (20 + i * 12, h // 2), 14, (obj, obj, obj), -1)
        cv2.imwrite(str(d / f"frame_{i:06d}.png"), f)
    return str(d)


def test_otsu_separates_dark_object():
    from count_cv import Detector
    frame = np.full((120, 200, 3), 180, np.uint8)          # bright background
    cv2.circle(frame, (100, 60), 16, (40, 40, 40), -1)     # darker object
    det = Detector({**DEFAULT_PARAMS, "method": "otsu"}, "otsu", 200, 120)
    mask = det._foreground(frame)
    assert mask[60, 100] == 255            # dark object -> foreground
    assert mask[5, 5] == 0                 # bright background -> not foreground
    assert det.last_threshold > 0          # Otsu chose a data-driven threshold


def test_otsu_counts_dark_object_crossing(tmp_path):
    from count_cv import count_source
    count = count_source(_dark_object_folder(tmp_path), params={"method": "otsu"}, fps=30.0)
    assert count == 1


def test_merge_close_preserves_multiplicity():
    from count_cv import Detector
    det = Detector({**DEFAULT_PARAMS, "merge_dist": 30.0}, "thresh", 200, 120)
    # Two near detections; one carries multiplicity 2 (unsplittable pair).
    dets = [(50, 60, 40, 50, 20, 20, 2), (60, 62, 50, 52, 20, 20, 1)]
    merged = det._merge_close(dets)
    assert len(merged) == 1
    assert merged[0][6] == 3          # 2 + 1 summed, not reset to 1
