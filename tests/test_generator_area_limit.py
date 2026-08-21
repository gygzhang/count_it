import math

import pytest

from gen_shapes_video import limit_size_range


def test_size_range_is_limited_to_twenty_percent_area_difference():
    effective_max = limit_size_range(10.0, 16.0)
    assert effective_max == pytest.approx(10.0 * math.sqrt(1.2))
    assert effective_max ** 2 / 10.0 ** 2 == pytest.approx(1.2)


def test_size_range_is_not_expanded_when_already_within_limit():
    assert limit_size_range(10.0, 10.5) == 10.5


@pytest.mark.parametrize("minimum,maximum", [(0, 1), (2, 1)])
def test_invalid_size_range_is_rejected(minimum, maximum):
    with pytest.raises(ValueError):
        limit_size_range(minimum, maximum)
