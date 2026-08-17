# Conveyor Counter Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superartes:subagent-driven-development (recommended) or superartes:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct tracking and merge semantics, add validated JSON-driven configuration and tuner preflight, split the counter into focused flat modules, and protect the behavior with pytest regression tests.

**Architecture:** Introduce `params.py`, `sources.py`, and `counting.py` with one-way dependencies; reduce `count_cv.py` to CLI orchestration and migrate every repository caller without compatibility exports. Preserve the current OpenCV pipeline while fixing only the confirmed semantics and validating all public entry points.

**Tech Stack:** Python 3, OpenCV, NumPy, argparse, concurrent.futures, pytest

**Authoritative spec:** `docs/specs/2026-08-17-conveyor-counter-hardening-design.md`

---

## File Map

- Create `params.py`: defaults, stage classifications, JSON loading, precedence, validation, ROI parsing.
- Create `sources.py`: natural sorting, video/image frame input, scaling, full decode.
- Create `counting.py`: foreground preparation, detection, tracking, counting, visualization output, ground-truth lookup.
- Modify `count_cv.py`: CLI parser and orchestration only; no compatibility exports.
- Modify `tune_params.py`: new imports, fixed `thresh` support, deterministic grid preflight.
- Modify `auto_params.py`: import `FrameSource` from `sources`.
- Modify `sort_compare.py`: import APIs from their owning modules.
- Create `requirements-dev.txt`: pytest development dependency.
- Create `tests/test_params.py`, `tests/test_sources.py`, `tests/test_tracker.py`, `tests/test_detector.py`, `tests/test_counting.py`, `tests/test_cli.py`, `tests/test_tuning.py`.
- Modify `README.md`: module layout, JSON config, tuner restrictions, consecutive-hit semantics, test instructions.

### Task 1: Establish pytest and central parameter model

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/test_params.py`
- Create: `params.py`
- Modify: `count_cv.py:42-68,143-148`
- Modify: `tune_params.py:26-28`

- [ ] **Step 1: Add the pytest development dependency**

Create `requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8.0
```

- [ ] **Step 2: Write failing parameter contract tests**

Create `tests/test_params.py` with focused tests equivalent to:

```python
import json

import pytest

from params import DEFAULT_PARAMS, load_params, merge_params, parse_roi, validate_params


def test_explicit_values_override_json_and_json_overrides_defaults(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"max_dist": 90, "min_hits": 3}), encoding="utf-8")
    merged = merge_params(load_params(path), {"max_dist": 180})
    assert merged["max_dist"] == 180
    assert merged["min_hits"] == 3
    assert merged["track_ttl"] == DEFAULT_PARAMS["track_ttl"]


def test_load_params_rejects_unknown_key(tmp_path):
    path = tmp_path / "params.json"
    path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_params(path)

def test_load_params_wraps_missing_file_as_runtime_error(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="missing.json"):
        load_params(path)


def test_load_params_reports_malformed_json_as_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_params(path)


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ({"scale": 0}, "scale"),
        ({"thresh_lo": 200, "thresh_hi": 100}, "thresh_lo"),
        ({"min_hits": 0}, "min_hits"),
        ({"max_dist": 0}, "max_dist"),
        ({"line": 1.1}, "line"),
        ({"line_band": 0.6}, "line_band"),
        ({"ref_alpha": -0.1}, "ref_alpha"),
    ],
)
def test_validate_params_rejects_invalid_values(change, name):
    params = {**DEFAULT_PARAMS, **change}
    with pytest.raises(ValueError, match=name):
        validate_params(params)


def test_parse_roi_clips_to_frame():
    assert parse_roi("-5,2,120,90", 100, 80) == (0, 2, 100, 80)


def test_parse_roi_rejects_empty_clipped_region():
    with pytest.raises(ValueError, match="roi"):
        parse_roi("120,2,140,20", 100, 80)
```

Also cover non-object JSON, enum values, non-negative numeric fields, fraction ranges, active min/max ordering, and ROI arity/type with separate named tests following the same assertion pattern.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_params.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'params'`.

- [ ] **Step 4: Implement `params.py` minimally**

Move `DEFAULT_PARAMS` out of `count_cv.py` without changing values. Define supported values and stage ownership:

```python
METHODS = ("auto", "color", "bgsub", "refbg", "thresh")
AXES = ("x", "y")
FLOWS = ("pos", "neg", "both")

PREPARATION_KEYS = {"method", "scale", "bg_ref"}
DETECTION_KEYS = {
    "method", "sat_thresh", "thresh_lo", "thresh_hi", "min_area",
    "max_area", "max_aspect", "min_area_frac", "max_area_frac",
    "morph_kernel", "morph_iter", "bg_history", "bg_var",
    "ref_thresh", "bg_ref", "ref_alpha", "split_area", "unit_area",
    "merge_dist", "roi", "scale", "axis",
}
TRACKING_KEYS = {
    "max_dist", "track_ttl", "min_hits", "min_speed", "line",
    "line_band", "axis", "flow", "warmup",
}
UNSEARCHABLE_GRID_KEYS = PREPARATION_KEYS | {"axis"}
```

Implement the public API with path-aware messages:

```python
def load_params(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise RuntimeError(f"unable to read params file: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in params file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"params file must contain a JSON object: {path}")
    unknown = sorted(set(data) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown parameter(s) in {path}: {', '.join(unknown)}")
    return data


def merge_params(file_params=None, cli_params=None):
    merged = dict(DEFAULT_PARAMS)
    merged.update(file_params or {})
    merged.update(cli_params or {})
    return validate_params(merged)


def parse_roi(roi, w, h):
    if roi is None:
        return None
    try:
        values = [int(v) for v in roi.split(",")] if isinstance(roi, str) else [int(v) for v in roi]
    except (TypeError, ValueError) as exc:
        raise ValueError("roi must contain four integers: x0,y0,x1,y1") from exc
    if len(values) != 4:
        raise ValueError("roi must contain four integers: x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    clipped = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"roi is empty after clipping to {w}x{h}: {roi}")
    return clipped
```

Implement `validate_params()` with the exact constraints in the spec. It must return the validated dictionary so callers can compose it. Do not coerce arbitrary strings from JSON; JSON values must already have the expected scalar type.

- [ ] **Step 5: Migrate existing constants and ROI parsing**

In `count_cv.py`, delete local `DEFAULT_PARAMS`, key sets, and `parse_roi`; import them from `params`. In `tune_params.py`, replace the old `DET_KEYS`/`TRK_KEYS` imports and every use at `eval_sample()` with `DETECTION_KEYS`/`TRACKING_KEYS`; do not add temporary aliases. Leave counting-function imports unchanged until Task 3.

- [ ] **Step 6: Run focused and syntax tests and verify GREEN**

Run:

```bash
pytest -q tests/test_params.py
python3 -m py_compile params.py count_cv.py tune_params.py
```

Expected: all parameter tests pass; compilation exits zero.

- [ ] **Step 7: Commit the releasable parameter checkpoint**

Invoke `superartes:commit-message`, then commit only the files in this task. Suggested subject: `Add validated parameter model`.

### Task 2: Extract and harden frame sources

**Files:**
- Create: `tests/test_sources.py`
- Create: `sources.py`
- Modify: `count_cv.py:30-123,412-428`
- Modify: `auto_params.py:16`
- Modify: `tune_params.py:26-28,40-76`

- [ ] **Step 1: Write failing source behavior tests**

Create `tests/test_sources.py`:

```python
import cv2
import numpy as np
import pytest

from sources import FrameSource, natural_key


def write_image(path, shape=(8, 12, 3)):
    assert cv2.imwrite(str(path), np.zeros(shape, dtype=np.uint8))


def test_natural_key_orders_numeric_names():
    names = ["frame10.jpg", "frame2.jpg", "frame1.jpg"]
    assert sorted(names, key=natural_key) == ["frame1.jpg", "frame2.jpg", "frame10.jpg"]


def test_directory_frame_rejects_dimension_mismatch(tmp_path):
    write_image(tmp_path / "frame1.jpg", (8, 12, 3))
    write_image(tmp_path / "frame2.jpg", (9, 12, 3))
    source = FrameSource(str(tmp_path))
    frames = source.frames()
    next(frames)
    with pytest.raises(RuntimeError, match="frame2.jpg.*12x8.*12x9"):
        next(frames)


def test_directory_frame_rejects_unreadable_image_when_read(tmp_path):
    write_image(tmp_path / "frame1.jpg")
    (tmp_path / "frame2.jpg").write_bytes(b"not an image")
    source = FrameSource(str(tmp_path))
    frames = source.frames()
    next(frames)
    with pytest.raises(RuntimeError, match="frame2.jpg"):
        next(frames)
```

Also test `sample()` applies the same readable/dimension checks.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `pytest -q tests/test_sources.py`.

Expected: collection fails because `sources.py` does not exist.

- [ ] **Step 3: Implement `sources.py`**

Move `IMG_EXTS`, `natural_key`, `FrameSource`, `scaled`, and `decode_all` from `count_cv.py`. Add one private reader used by both directory access paths:

```python
def _read_image(self, path):
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"unable to read image: {path}")
    actual_h, actual_w = image.shape[:2]
    if (actual_w, actual_h) != (self.w, self.h):
        raise RuntimeError(
            f"image size mismatch: {path}; expected {self.w}x{self.h}, "
            f"got {actual_w}x{actual_h}"
        )
    return image
```

The constructor reads and validates the first image. `frames()` and `sample()` call `_read_image()` for directory inputs; they no longer silently skip unreadable files. Preserve video behavior and explicit `release()`.

- [ ] **Step 4: Migrate frame-source callers**

Delete moved definitions from `count_cv.py` and import from `sources`. Change:

```python
# auto_params.py
from sources import FrameSource

# tune_params.py
from sources import FrameSource, decode_all, scaled
```

Keep each module's other current imports until Task 4 performs the clean counting cutover.

- [ ] **Step 5: Run focused and existing checks and verify GREEN**

Run:

```bash
pytest -q tests/test_sources.py tests/test_params.py
python3 -m py_compile sources.py count_cv.py auto_params.py tune_params.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 6: Commit the releasable source checkpoint**

Invoke `superartes:commit-message`. Suggested subject: `Validate frame sequence inputs`.

### Task 3: Extract counting and correct tracker and merge semantics

**Files:**
- Create: `tests/test_tracker.py`
- Create: `tests/test_detector.py`
- Create: `tests/test_counting.py`
- Create: `counting.py`
- Modify: `count_cv.py:126-565`
- Modify: `tune_params.py:26-28`
- Modify: `sort_compare.py:12-13`

- [ ] **Step 1: Write the failing consecutive-hit regression**

Create `tests/test_tracker.py` using a helper that returns a complete parameter dictionary. The regression must prove that a gap restarts confirmation:

```python
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
```

Add focused tests for reacquisition within TTL, positive/negative flow, hysteresis, one count per track, and TTL expiry.

- [ ] **Step 2: Run the tracker regression and verify RED**

Run `pytest -q tests/test_tracker.py::test_missed_frame_restarts_consecutive_confirmation`.

Expected: collection fails because `counting.py` does not exist. If extraction has begun locally, the assertion must fail because current `hits` accumulates across the gap.

- [ ] **Step 3: Create `counting.py` and implement the minimal tracker correction**

Move method preparation, `Track`, `Detector`, `Tracker`, sequence helpers, `count_source`, and `find_gt` from `count_cv.py`. Import source and parameter APIs from their owning modules.

Replace cumulative hits with:

```python
class Track:
    def __init__(self, track_id, cx, cy):
        self.id = track_id
        self.cx, self.cy = cx, cy
        self.vx, self.vy = 0.0, 0.0
        self.prev_main = cx
        self.min_main = float("inf")
        self.max_main = float("-inf")
        self.consecutive_hits = 1
        self.missing = 0
        self.counted = False
        self.matched = False
```

Each `Tracker.__init__()` sets `self._next_id = 0`. A private track-creation helper passes the current value into `Track(track_id, cx, cy)` and then increments it, so every Tracker instance starts at ID 0 without class-global state. A successful match increments `consecutive_hits` when it is already positive and sets it to 1 after a miss reset it to zero. An unmatched frame resets it to zero while preserving `missing`, velocity coasting, and TTL behavior. Counting gates on `consecutive_hits`.

- [ ] **Step 4: Run all tracker tests and verify GREEN**

Run `pytest -q tests/test_tracker.py`.

Expected: all tracker contracts pass.

- [ ] **Step 5: Write the failing transitive merge regression**

Create `tests/test_detector.py`:

```python
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
```

Use the exact expected numeric tuple produced by the implemented bounding-box convention; the connection boundary remains strict `< merge_dist`.

- [ ] **Step 6: Run the merge regression and verify RED**

Run `pytest -q tests/test_detector.py::test_merge_close_is_transitive_and_order_independent`.

Expected: current one-pass grouping returns two groups for at least one ordering.

- [ ] **Step 7: Implement union-find grouping minimally**

Use pairwise distance checks and union-find. Group members by root, compute the union bounding rectangle, derive its center, and sort groups by their minimum original index. Do not add a spatial index.

- [ ] **Step 8: Verify detector and tracker GREEN**

Run:

```bash
pytest -q tests/test_detector.py tests/test_tracker.py
```

Expected: all tests pass.

- [ ] **Step 9: Complete the clean counting cutover**

Delete moved implementations from `count_cv.py`; import only `count_source` and `find_gt` for CLI use. Migrate:

```python
# tune_params.py
from counting import count_source, detect_sequence, find_gt, resolve_method, track_sequence

# sort_compare.py
from counting import Detector, Tracker, find_gt, is_warming, prepare_method
from params import DEFAULT_PARAMS
from sources import FrameSource, scaled
```

Do not re-export old symbols from `count_cv.py`. Verify with a test that two independently constructed `Tracker` instances both assign ID 0 to their first detection.

- [ ] **Step 10: Add failing programmatic validation regression**

Create `tests/test_counting.py` with a temporary 10×10 image directory and call the public API directly:

```python
import cv2
import numpy as np
import pytest

from counting import count_source


def test_count_source_validates_roi_against_processed_dimensions(tmp_path):
    assert cv2.imwrite(str(tmp_path / "frame1.jpg"), np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="roi"):
        count_source(str(tmp_path), {"roi": "20,0,30,5"})
```

Run `pytest -q tests/test_counting.py::test_count_source_validates_roi_against_processed_dimensions` and verify RED: current code reaches OpenCV with an empty crop instead of raising the required `ValueError`.

- [ ] **Step 11: Validate inside `count_source()` and verify GREEN**

At the start of `count_source()`, call `merge_params(cli_params=params or {})` to create and dimension-independently validate the complete parameter dictionary. After opening `FrameSource` and computing the scaled `w` and `h`, call `validate_params(P, w, h)` before method preparation, detector construction, writer creation, or frame iteration. Run the focused test again; expected: PASS with a path-independent configuration `ValueError`.

- [ ] **Step 12: Add writer-open regression and validation**

In `tests/test_counting.py`, monkeypatch only `cv2.VideoWriter` with an object whose `isOpened()` returns false, invoke `count_source()` on a one-frame temporary image directory, and assert a destination-specific `RuntimeError` occurs before `frames()` is consumed. Add the `isOpened()` check immediately after constructing the writer.

Run the focused test first to observe RED, implement, then rerun it to GREEN.

- [ ] **Step 13: Run the suite and syntax checks**

Run:

```bash
pytest -q
python3 -m py_compile counting.py count_cv.py tune_params.py sort_compare.py
```

Expected: all tests pass; compilation exits zero.

- [ ] **Step 14: Commit the releasable counting checkpoint**

Invoke `superartes:commit-message`. Suggested subject: `Correct counting track semantics`.

### Task 4: Add JSON CLI precedence and tuner preflight

**Files:**
- Create or complete: `tests/test_cli.py`
- Create: `tests/test_tuning.py`
- Modify: `count_cv.py:568-641`
- Modify: `tune_params.py:79-249`
- Modify: `params.py`

- [ ] **Step 1: Write failing CLI precedence tests**

Expose a small testable function such as `resolve_cli_params(argv)` in `count_cv.py`. Test the observable CLI contract without running a large video:

```python
import json

import pytest

from count_cv import resolve_cli_params


def test_json_values_survive_when_cli_option_is_absent(tmp_path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"max_dist": 91, "min_hits": 3}), encoding="utf-8")
    source, params, outputs = resolve_cli_params(["input.mp4", "--params", str(path)])
    assert source == "input.mp4"
    assert params["max_dist"] == 91
    assert params["min_hits"] == 3


def test_explicit_cli_value_overrides_json(tmp_path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"max_dist": 91}), encoding="utf-8")
    _, params, _ = resolve_cli_params(
        ["input.mp4", "--params", str(path), "--max-dist", "180"]
    )
    assert params["max_dist"] == 180


def test_invalid_json_parameter_becomes_argparse_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"scale": 0}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        resolve_cli_params(["input.mp4", "--params", str(path)])
    assert exc.value.code == 2


def test_missing_params_file_becomes_argparse_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        resolve_cli_params(["input.mp4", "--params", str(tmp_path / "missing.json")])
    assert exc.value.code == 2


def test_malformed_params_file_becomes_argparse_error(tmp_path):
    path = tmp_path / "bad-json.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        resolve_cli_params(["input.mp4", "--params", str(path)])
    assert exc.value.code == 2
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run `pytest -q tests/test_cli.py -k 'json or explicit'`.

Expected: `resolve_cli_params` or `--params` is missing.

- [ ] **Step 3: Implement explicit CLI precedence**

Add `--params`. Parameter options must use `default=argparse.SUPPRESS`; output-only options retain normal defaults. `resolve_cli_params(argv)` must:

1. Parse arguments.
2. Load the JSON file if supplied.
3. Build CLI overrides only from parameter attributes present in the namespace.
4. Call `merge_params()`.
5. Convert `ValueError` and `RuntimeError` from configuration loading/validation into `parser.error(str(exc))`.
6. Return the source, complete parameters, and output options used by `main()`.

Do not keep `args_to_params()` if it can no longer express explicitness correctly.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run `pytest -q tests/test_cli.py`.

Expected: precedence, error rendering, and writer tests pass.

- [ ] **Step 5: Write failing tuner parser and grid tests**

Refactor `tune_params.py` to expose `build_arg_parser()` and `parse_grid(raw, base, samples)` as public test seams. Create `tests/test_tuning.py`:

```python
import pytest

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


@pytest.mark.parametrize("raw", ["{}", '{"max_dist": []}', '{"max_dist": 10}', '{"unknown": [1]}'])
def test_grid_rejects_invalid_structure(raw):
    with pytest.raises(ValueError):
        parse_grid(raw, merge_params(), [])
```

Add a temporary-image sample test proving an ROI candidate that clips to empty is rejected before a supplied worker/evaluator callback can run.

- [ ] **Step 6: Run tuner tests and verify RED**

Run `pytest -q tests/test_tuning.py`.

Expected: `thresh` is rejected by the current parser and grid preflight functions are missing.

- [ ] **Step 7: Implement deterministic tuner preflight**

Move parser construction into `build_arg_parser()` and include `thresh` in choices. `parse_grid()` must:

- Decode JSON and require a non-empty object.
- Reject unknown and `UNSEARCHABLE_GRID_KEYS` keys.
- Require each value to be a non-empty list.
- Expand every combination in deterministic key/list order.
- Merge each combination with the fixed base and run dimension-independent validation.
- Name the key, candidate value, and combination index in errors.

Before `ProcessPoolExecutor`, `FrameCache`, or sequence decoding, inspect each sample's source metadata using the fixed scale and call `validate_params(candidate, w, h)` for every candidate. Release each source immediately. This handles ROI empty-after-clipping deterministically in the main process.

Use `DETECTION_KEYS` and `TRACKING_KEYS` only after preflight. Because `axis` is rejected, no cross-stage candidate reaches the cache loops.

- [ ] **Step 8: Run focused and full tests and verify GREEN**

Run:

```bash
pytest -q tests/test_tuning.py tests/test_cli.py
pytest -q
```

Expected: all tests pass.

- [ ] **Step 9: Run actual JSON and tuning smoke tests**

Run:

```bash
python3 count_cv.py simp.mp4 --params best_params.json --method bgsub --warmup 8
python3 tune_params.py samples.txt --grid '{"min_area":[60,150],"max_dist":[80,120]}' --topk 4 --out ~/tmp/inv_best_params.json
```

Expected: counting command completes through JSON loading; the tuning run evaluates four combinations and reports training `SAE=0`.

- [ ] **Step 10: Commit the releasable configuration checkpoint**

Invoke `superartes:commit-message`. Suggested subject: `Complete parameter tuning workflow`.

### Task 5: Update documentation and perform final verification

**Files:**
- Modify: `README.md`
- Verify: all Python files and CLI workflows

- [ ] **Step 1: Update README module and configuration documentation**

Document:

```text
sources.py   video/image sequence input
params.py    defaults, JSON loading, validation
counting.py  detector, tracker, count_source
count_cv.py  command-line entry point only
```

Add installation:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # contributors/tests
```

Add JSON use and precedence:

```bash
python3 count_cv.py clip.mp4 --params best_params.json
python3 count_cv.py clip.mp4 --params best_params.json --max-dist 180
```

State `DEFAULT_PARAMS < JSON < explicit CLI`, `min_hits` means consecutive observations, `thresh` is a valid fixed tuner method, and `method`, `scale`, `bg_ref`, and `axis` cannot be grid-searched under the current cache model.

Update programmatic examples to import from `sources`, `params`, and `counting`; remove any claim that core symbols come from `count_cv`.

- [ ] **Step 2: Run documentation-sensitive CLI help checks**

Run:

```bash
python3 count_cv.py --help
python3 tune_params.py --help
```

Expected: `count_cv.py` lists `--params`; tuner method choices include `thresh`; help exits zero without tracebacks.

- [ ] **Step 3: Run the full automated suite**

Run:

```bash
pytest -q
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 4: Compile every executable and module**

Run:

```bash
python3 -m py_compile count_cv.py counting.py params.py sources.py tune_params.py auto_params.py gen_shapes_video.py video_to_frames.py sort_compare.py
```

Expected: exit zero with no output.

- [ ] **Step 5: Run the actual counter smoke test**

Run:

```bash
python3 count_cv.py simp.mp4 --method bgsub --warmup 8
```

Expected:

```text
CV 计数结果: 4
真值(越过中线): 4  |  误差: 0
```

- [ ] **Step 6: Run the actual tracker comparison smoke test**

Run:

```bash
python3 sort_compare.py simp.mp4 --max-dist 40
```

Expected: custom Tracker and simplified SORT both report 4 against truth 4.

- [ ] **Step 7: Run final diff hygiene check**

Run `git diff --check`.

Expected: exit zero. Confirm the pre-existing untracked `sort_compare.py` is handled as user work: implementation may edit it because it is an explicit migration target, but must not stage unrelated media or temporary artifacts.

- [ ] **Step 8: Commit the documentation and verified cutover**

Invoke `superartes:commit-message`. Suggested subject: `Document hardened counter workflow`.

## Completion Criteria

- No repository file imports core APIs from `count_cv.py`.
- `count_cv.py` contains CLI parsing/orchestration only.
- No compatibility aliases or deprecated exports remain.
- Every specified test passes and each new behavioral test was observed failing before implementation.
- JSON config, explicit CLI overrides, fixed `thresh` tuning, and deterministic grid rejection work through actual entry points.
- Existing `simp.mp4` counting remains exactly `4/4`; the small tuning grid remains `SAE=0`.
