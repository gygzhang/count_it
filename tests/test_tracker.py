from counting import Tracker
from params import merge_params


def det(x, y=5):
    return (x, y, x - 1, y - 1, 2, 2)


def tracker_params(**changes):
    return merge_params(cli_params={
        "axis": "x", "flow": "pos", "line": 0.5,
        "max_dist": 10, "track_ttl": 5, **changes,
    })


def test_missed_frame_restarts_consecutive_confirmation():
    tracker = Tracker(tracker_params(min_hits=2), 10, 10)
    tracker.update([det(4)])
    tracker.update([])
    tracker.update([det(6)])
    assert tracker.tracks[0].consecutive_hits == 1
    assert tracker.count == 0


def test_reacquisition_within_ttl_retains_trajectory_but_reconfirms():
    tracker = Tracker(tracker_params(min_hits=2), 10, 10)
    tracker.update([det(2)])
    track_id = tracker.tracks[0].id
    tracker.update([])
    tracker.update([det(3)])
    assert tracker.tracks[0].id == track_id
    assert tracker.tracks[0].consecutive_hits == 1
    tracker.update([det(4)])
    assert tracker.tracks[0].consecutive_hits == 2


def test_positive_and_negative_flow_gate_crossings():
    positive = Tracker(tracker_params(flow="pos"), 10, 10)
    positive.update([det(3)])
    positive.update([det(6)])
    assert positive.count == 1

    negative = Tracker(tracker_params(flow="neg"), 10, 10)
    negative.update([det(7)])
    negative.update([det(4)])
    assert negative.count == 1

    blocked = Tracker(tracker_params(flow="pos"), 10, 10)
    blocked.update([det(7)])
    blocked.update([det(4)])
    assert blocked.count == 0


def test_hysteresis_requires_starting_beyond_line_band():
    tracker = Tracker(tracker_params(line_band=0.2), 10, 10)
    tracker.update([det(4)])
    tracker.update([det(6)])
    assert tracker.count == 0

    tracker = Tracker(tracker_params(line_band=0.2), 10, 10)
    tracker.update([det(3)])
    tracker.update([det(6)])
    assert tracker.count == 1


def test_track_is_counted_once():
    tracker = Tracker(tracker_params(flow="both"), 10, 10)
    tracker.update([det(3)])
    tracker.update([det(6)])
    tracker.update([det(3)])
    tracker.update([det(6)])
    assert tracker.count == 1


def test_track_expires_after_ttl():
    tracker = Tracker(tracker_params(track_ttl=2), 10, 10)
    tracker.update([det(2)])
    track_id = tracker.tracks[0].id
    tracker.update([])
    tracker.update([])
    assert tracker.tracks
    tracker.update([])
    assert not tracker.tracks
    tracker.update([det(2)])
    assert tracker.tracks[0].id != track_id


def test_tracker_ids_are_instance_local():
    first = Tracker(tracker_params(), 10, 10)
    second = Tracker(tracker_params(), 10, 10)
    first.update([det(2)])
    second.update([det(2)])
    assert first.tracks[0].id == 0
    assert second.tracks[0].id == 0
