import cv2
import numpy as np

import annotate


def make_folder(tmp_path, n=14, bg=120, obj=220, w=200, h=120):
    """A uniform gray belt with one bright object crossing the centre line."""
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        f = np.full((h, w, 3), bg, np.uint8)
        cv2.circle(f, (20 + i * 12, h // 2), 14, (obj, obj, obj), -1)
        cv2.imwrite(str(d / f"frame_{i:06d}.png"), f)
    return str(d)


def test_annotate_source_counts_and_structure(tmp_path):
    result = annotate.annotate_source(make_folder(tmp_path), fps=30.0)

    assert result["resolved"]["method"] == "thresh"     # uniform gray -> thresh
    assert result["count"] == 1                         # object crosses once
    assert result["meta"]["frames"] == 14
    assert len(result["frames"]) == 14

    w, h = result["meta"]["width"], result["meta"]["height"]
    for fr in result["frames"]:
        assert set(fr) >= {"i", "dets", "tracks", "events", "count"}
        for x, y, bw, bh in fr["dets"]:
            assert 0 <= x <= w and 0 <= y <= h and bw > 0 and bh > 0
    # exactly one crossing event fires across the whole run
    assert sum(len(fr["events"]) for fr in result["frames"]) == 1
    assert result["frames"][-1]["count"] == 1


def test_annotate_dumps_processed_frames(tmp_path):
    out = tmp_path / "jpg"
    annotate.annotate_source(make_folder(tmp_path, n=6), fps=30.0,
                             dump_frames=str(out))
    assert sorted(p.name for p in out.iterdir()) == [
        f"frame_{i:06d}.jpg" for i in range(6)]


def test_frame_metrics_matches_truth(tmp_path):
    result = annotate.annotate_source(make_folder(tmp_path), fps=30.0)
    per = {i: [[20 + i * 12 - 14, 120 // 2 - 14, 28, 28]] for i in range(14)}
    truth = {"gt_total": 1, "per_frame": per}

    m = annotate.frame_metrics(result, truth)
    assert m["count"] == 1 and m["gt_total"] == 1 and m["count_error"] == 0
    assert m["tp"] >= 10 and m["recall"] > 0.7


def test_frame_metrics_without_per_frame_truth(tmp_path):
    result = annotate.annotate_source(make_folder(tmp_path), fps=30.0)
    m = annotate.frame_metrics(result, {"gt_total": 1, "per_frame": None})
    assert m["count_error"] == 0
    assert "tp" not in m         # no per-frame boxes -> only count-vs-GT
