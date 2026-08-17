from counting import Detector
from params import merge_params


def make_detector():
    params = merge_params(cli_params={"method": "thresh", "merge_dist": 5})
    return Detector(params, "thresh", 100, 100)


def test_merge_close_is_transitive_and_order_independent():
    detector = make_detector()
    a = (0, 0, 0, 0, 2, 2)
    c = (8, 0, 8, 0, 2, 2)
    b = (4, 0, 4, 0, 2, 2)
    assert detector._merge_close([a, c, b]) == [(5, 1, 0, 0, 10, 2)]
    assert detector._merge_close([c, b, a]) == [(5, 1, 0, 0, 10, 2)]


def test_merge_close_uses_strict_distance_boundary():
    detector = make_detector()
    left = (0, 0, 0, 0, 2, 2)
    right = (5, 0, 5, 0, 2, 2)
    assert detector._merge_close([left, right]) == [
        (1.0, 1.0, 0, 0, 2, 2), (6.0, 1.0, 5, 0, 2, 2)
    ]

def test_merge_close_orders_disconnected_components_geometrically():
    detector = make_detector()
    left = (20, 0, 20, 0, 2, 2)
    right = (0, 0, 0, 0, 2, 2)
    expected = [(1.0, 1.0, 0, 0, 2, 2), (21.0, 1.0, 20, 0, 2, 2)]
    assert detector._merge_close([left, right]) == expected
    assert detector._merge_close([right, left]) == expected
