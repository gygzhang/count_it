import cv2, numpy as np
from count_cv import DEFAULT_PARAMS, Detector

def test_watershed_splits_touching_blobs():
    p = dict(DEFAULT_PARAMS)
    p.update(method='thresh', thresh_lo=100, thresh_hi=150,
             min_area=20, morph_kernel=1, morph_iter=1,
             watershed_split=True, watershed_min_distance=8)
    # two bright circles touching at a narrow bridge on gray background
    g = np.full((100, 180), 120, np.uint8)
    cv2.circle(g, (65, 50), 20, 230, -1)
    cv2.circle(g, (95, 50), 20, 230, -1)
    frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    dets = Detector(p, 'thresh', 180, 100).detect(frame)
    assert len(dets) == 2
