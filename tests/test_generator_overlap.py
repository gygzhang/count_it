from types import SimpleNamespace

import numpy as np

from gen_shapes_video import object_overlap_ratio


def _square(x, y, size=10):
    return SimpleNamespace(
        x=float(x),
        y=float(y),
        asset_alpha=np.full((size, size), 255, np.uint8),
    )


def test_overlap_ratio_uses_smaller_foreground_area():
    # Two 10x10 masks with one shared column: 10/100 = 10%.
    assert object_overlap_ratio(_square(0, 0), _square(9, 0)) == 0.1


def test_non_intersecting_objects_have_zero_overlap():
    assert object_overlap_ratio(_square(0, 0), _square(11, 0)) == 0.0


def test_five_percent_overlap_boundary():
    # 20x20 masks with one shared column: 20/400 = 5%.
    ratio = object_overlap_ratio(_square(0, 0, 20), _square(19, 0, 20))
    assert ratio == 0.05

