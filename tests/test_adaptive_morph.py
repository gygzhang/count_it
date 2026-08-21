import cv2
import numpy as np

from count_cv import DEFAULT_PARAMS, Detector


def test_morphology_kernel_shrinks_with_processing_scale():
    params = {
        **DEFAULT_PARAMS,
        "method": "color",
        "scale": 0.25,
        "morph_kernel": 7,
    }
    detector = Detector(params, "color", 200, 125)

    assert detector.kernel.shape == (1, 1)


def test_adaptive_morph_preserves_small_scaled_object():
    frame = np.zeros((125, 200, 3), dtype=np.uint8)
    # Saturated 3 px-wide object: a 7x7 opening would erase it.
    cv2.rectangle(frame, (90, 50), (92, 65), (0, 0, 255), -1)
    params = {
        **DEFAULT_PARAMS,
        "method": "color",
        "scale": 0.25,
        "sat_thresh": 20,
        "min_area": 2,
        "morph_kernel": 7,
    }

    assert len(Detector(params, "color", 200, 125).detect(frame)) == 1

    params["adaptive_morph"] = False
    assert Detector(params, "color", 200, 125).detect(frame) == []
