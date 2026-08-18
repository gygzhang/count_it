import json

import cv2
import numpy as np
import pytest

import tune_params
from params import merge_params
from tune_params import build_arg_parser, parse_grid


def test_parser_accepts_thresh_as_fixed_method():
    args = build_arg_parser().parse_args(["samples.txt", "--method", "thresh"])
    assert args.method == "thresh"


@pytest.mark.parametrize("key", ["method", "scale", "bg_ref", "axis"])
def test_grid_rejects_cache_unsafe_key(key):
    raw = '{"%s": [1]}' % key
    with pytest.raises(ValueError, match=key):
        parse_grid(raw, merge_params(), [])


@pytest.mark.parametrize(
    "raw",
    ["{}", '{"max_dist": []}', '{"max_dist": 10}', '{"unknown": [1]}'],
)
def test_grid_rejects_invalid_structure(raw):
    with pytest.raises(ValueError):
        parse_grid(raw, merge_params(), [])


def test_grid_order_is_deterministic_and_all_combinations_are_valid():
    grid = parse_grid(
        '{"max_dist": [80, 120], "min_hits": [1, 2]}',
        merge_params(),
        [],
    )

    assert list(grid) == ["max_dist", "min_hits"]
    assert grid["max_dist"] == [80, 120]
    assert grid["min_hits"] == [1, 2]


def test_empty_clipped_roi_is_rejected_before_worker_dispatch(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    manifest = tmp_path / "samples.txt"
    manifest.write_text(f"{image_path},0\n")

    def should_not_run(_task):
        raise AssertionError("worker dispatched before parameter preflight")

    monkeypatch.setattr(tune_params, "eval_sample", should_not_run)
    with pytest.raises(SystemExit) as exc:
        tune_params.main(
            [
                str(manifest),
                "--grid",
                json.dumps({"roi": ["100,0,120,10"]}),
            ]
        )

    assert exc.value.code == 2
