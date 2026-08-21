from count_cv import DEFAULT_PARAMS, Tracker


def _det(x, y=20):
    return (float(x), float(y), float(x - 2), float(y - 2), 4.0, 4.0)


def test_global_velocity_and_order_keep_close_fast_objects_separate():
    params = {
        **DEFAULT_PARAMS,
        "axis": "x", "flow": "pos", "line": 0.5,
        "max_dist": 70.0, "track_ttl": 2, "min_hits": 1,
        "global_vx": 60.0, "global_vy": 0.0, "ordered_match": True,
    }
    tracker = Tracker(params, 200, 40)
    # Two objects are only 10 px apart but both move 60 px/frame.
    for xs in ([10, 20], [70, 80], [130, 140]):
        tracker.update([_det(x) for x in xs])

    assert tracker.count == 2
    active = [t for t in tracker.tracks if t.missing == 0]
    assert sorted(round(t.vx) for t in active) == [60, 60]


def test_transverse_gate_keeps_vertical_lanes_separate():
    params = {
        **DEFAULT_PARAMS,
        "axis": "y", "flow": "pos", "line": 0.5,
        "max_dist": 80.0, "track_ttl": 2, "min_hits": 1,
        "global_vx": 0.0, "global_vy": 50.0, "ordered_match": True,
        "transverse_gate": 8.0,
    }
    tracker = Tracker(params, 100, 200)
    # Two lanes are close in the motion-axis ordering, but remain distinct in x.
    for ys in ([10, 12], [60, 62], [110, 112]):
        tracker.update([_det(20, ys[0]), _det(50, ys[1])])
    assert tracker.count == 2
    active = sorted((round(t.cx), round(t.vy))
                    for t in tracker.tracks if t.missing == 0)
    assert active == [(20, 50), (50, 50)]


def test_area_gate_rejects_wrong_nearby_detection():
    params = {
        **DEFAULT_PARAMS,
        "axis": "x", "flow": "pos", "line": 0.5,
        "max_dist": 50.0, "ordered_match": False,
        "area_ratio_max": 1.5, "shape_cost_weight": 2.0,
        "global_vx": 20.0,
    }
    tracker = Tracker(params, 200, 100)
    tracker.update([(20.0, 20.0, 15.0, 15.0, 10.0, 10.0)])
    # A much larger false blob is closer than the true equal-size object.
    tracker.update([
        (38.0, 20.0, 18.0, 0.0, 40.0, 40.0),
        (40.0, 20.0, 35.0, 15.0, 10.0, 10.0),
    ])
    original = min(tracker.tracks, key=lambda t: t.id)
    assert round(original.cx) == 40


def test_crossing_dedup_suppresses_split_track_event():
    params = {
        **DEFAULT_PARAMS,
        "axis": "x", "flow": "pos", "line": 0.5,
        "max_dist": 50.0, "track_ttl": 2, "min_hits": 1,
        "global_vx": 20.0, "ordered_match": False,
        "cross_dedup_frames": 2, "cross_dedup_dist": 3.0,
    }
    tracker = Tracker(params, 100, 40)
    # Two detections on the same image row cross the line in adjacent frames;
    # this mimics a watershed split of one physical object.
    tracker.update([_det(40, 20)])
    tracker.update([_det(60, 20), _det(61, 20)])
    assert tracker.count == 1
