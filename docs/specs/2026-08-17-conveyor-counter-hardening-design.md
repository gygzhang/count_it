# Conveyor Counter Hardening Design

**Date:** 2026-08-17

## Goal

Correct the confirmed tracking and detection defects, complete the JSON parameter and tuning workflow, split the monolithic counter into focused modules, and add deterministic regression coverage without changing the supported conveyor-counting domain.

## Scope

This change includes:

- Consecutive-hit semantics for `min_hits`.
- Order-independent transitive detection merging.
- JSON parameter loading with explicit CLI override precedence.
- Central parameter and runtime-input validation.
- `thresh` support and safe grid validation in the tuner.
- A clean module cutover to `sources.py`, `params.py`, and `counting.py`.
- Pytest regression tests and updated user documentation.

This change does not add new detection algorithms, multi-lane tracking, occlusion recovery, services, GUIs, model training, or compatibility exports from `count_cv.py`.

## Architecture

The repository remains a flat Python script project. Responsibilities move into three focused modules:

```text
sources.py
  FrameSource, natural_key, scaled, decode_all
        |
        v
params.py
  DEFAULT_PARAMS, parameter classifications, JSON loading,
  merge precedence, validation, ROI parsing
        |
        v
counting.py
  foreground-method preparation, Detector, Track, Tracker,
  sequence helpers, count_source, find_gt
        |
        v
count_cv.py
  argparse construction, --params handling, CLI orchestration,
  result and ground-truth output
```

Repository callers migrate to the new ownership boundaries:

- `auto_params.py` imports frame input from `sources`.
- `tune_params.py` imports source, parameter, and counting APIs from their owning modules.
- `sort_compare.py` imports from all three modules as needed.
- `count_cv.py` is CLI-only and no longer provides `Detector`, `Tracker`, or other historical import symbols.

Dependencies remain acyclic. No package hierarchy, registries, inheritance framework, or compatibility aliases are introduced.

## Parameter Data Flow

Final parameters use this precedence:

```text
DEFAULT_PARAMS < --params JSON < explicitly supplied CLI options
```

Argparse defaults must not overwrite JSON values. The CLI therefore records only explicitly supplied parameter options and passes those overrides to `merge_params()`.

`params.py` owns:

- `DEFAULT_PARAMS`.
- Parameter stage classifications used by the tuner.
- `load_params(path)`: read a JSON object and reject unknown keys.
- `merge_params(file_params, cli_params)`: produce a complete parameter dictionary using the documented precedence.
- `validate_params(params, width=None, height=None)`: validate values and cross-field constraints.
- `parse_roi(roi, width, height)`: accept a four-integer string or sequence, clip it to the processed frame, and reject an empty result.

`count_source()` validates its complete parameter dictionary as well as the CLI. Programmatic callers therefore cannot bypass validation.

Validation covers:

- `scale > 0`.
- `thresh_lo` and `thresh_hi` are within 0–255 and ordered.
- Pixel areas, speed, TTL, warmup, morphology iterations, and related counts are non-negative.
- `min_hits >= 1` and `max_dist > 0`.
- `line` is within 0–1 and `line_band` within 0–0.5.
- Area fractions and `ref_alpha` are within 0–1.
- If both minimum and maximum limits are active, maximum is not below minimum.
- ROI contains exactly four integers and is non-empty after clipping.
- Enum values match the supported method, axis, and flow sets.
- Unknown JSON keys are rejected rather than ignored.

Configuration errors raise `ValueError`. File, frame, decoder, and writer failures raise `RuntimeError`. The CLI converts configuration errors to `argparse.error()` messages; programmatic APIs preserve exceptions.

## Tracking Correction

`Track.hits` becomes `Track.consecutive_hits`:

- A new track starts at 1.
- A successful detection match increments it.
- Any unmatched frame resets it to 0.
- `missing`, velocity-based coasting, and TTL retention remain unchanged.
- The counting gate compares `consecutive_hits` with `min_hits`.

This makes `min_hits` match its documented meaning: a track must be observed in the required number of consecutive frames before it can count. Reacquisition within TTL continues the same trajectory but starts confirmation again.

The existing crossing rules remain unchanged:

- Direction is controlled by `flow`.
- `line_band` requires the trajectory to originate beyond the hysteresis band.
- `min_speed` gates axis velocity.
- A trajectory counts at most once.
- MOG2 warmup suppresses counting but continues detector and tracker state updates.

## Detection Merge Correction

`Detector._merge_close()` becomes an order-independent connected-components operation:

1. Treat every detection as a node.
2. Connect two nodes when their center distance is less than `merge_dist`.
3. Compute transitive groups with union-find.
4. Replace each group with the bounding rectangle covering every member.
5. Emit groups in a geometry-derived stable order `(x0, y0, x1, y1)`; use minimum original index only as a final tie-breaker for exactly coincident boxes. This makes output independent of contour enumeration while preserving deterministic results for identical geometry.

This ensures an `A-B-C` chain merges into one group when `A` is close to `B` and `B` is close to `C`, regardless of contour order.

The initial implementation may examine all detection pairs because merge mode is optional and the normal expected detection count is small. It must not add a spatial index unless profiling shows this operation matters.

## Source and Output Validation

`FrameSource` verifies directory images lazily when each file is read by `frames()` or `sample()`. Every image must decode successfully and have the same dimensions as the first frame. An unreadable or mismatched frame raises `RuntimeError` naming the file and, for mismatches, the expected and actual dimensions. The constructor still validates the first image immediately; it does not eagerly decode the whole directory.

Video or image decode failures retain path-specific errors. When annotated video output is requested, `count_source()` checks `VideoWriter.isOpened()` before processing frames and raises `RuntimeError` if the destination or codec cannot be opened.

## Tuning Contract

`tune_params.py --method` accepts:

- `auto`
- `color`
- `bgsub`
- `refbg`
- `thresh`

A custom grid must be a non-empty JSON object whose keys are known, searchable parameters and whose values are non-empty lists. Before any `ProcessPoolExecutor`, frame cache, or sequence decoding starts, the main process expands every combination and validates each complete parameter dictionary. Dimension-independent errors identify the grid key, candidate value, and combination index. ROI syntax is validated in this pass; because empty-after-clipping depends on each sample's processed dimensions, the main process also reads source metadata and validates every ROI candidate against every sample before dispatching workers. Any invalid candidate aborts the run deterministically before evaluation.

The current cache architecture cannot correctly search parameters that affect decode/preparation or both detector and tracker stages. The tuner rejects these grid keys with a clear error:

- `method`
- `scale`
- `bg_ref`
- `axis`

`axis` is intentionally fixed-only because it controls both `Detector`'s direction for splitting a large connected component and `Tracker`'s counting coordinate. The current `TRK_KEYS` classification is incomplete: changing `axis` only in the tracker combination would incorrectly reuse detections split along the base axis. The other rejected keys affect frame dimensions or foreground-method/reference preparation before the detector-combination loop.

They remain supported as fixed CLI/base parameters. Other detector parameters reuse detection results by detector combination; tracker-only combinations reuse those detection sequences.

`tune_params.py` continues writing a complete parameter object. That file becomes directly usable through:

```bash
python3 count_cv.py SOURCE --params best_params.json
```

Explicit options override the file, for example:

```bash
python3 count_cv.py SOURCE --params best_params.json --max-dist 180
```

## Tests

Pytest is a development dependency in `requirements-dev.txt`; runtime dependencies remain in `requirements.txt`.

Tests use temporary directories and small NumPy/OpenCV images rather than repository video artifacts:

- `tests/test_tracker.py`
  - A miss resets consecutive confirmation.
  - Reacquisition within TTL retains the track but requires confirmation again.
  - Positive and negative direction gates.
  - Hysteresis behavior.
  - A track counts at most once.
  - TTL expiry.

- `tests/test_detector.py`
  - Transitive merge of an `A-B-C` chain.
  - Identical grouping under reordered input.
  - Correct bounding rectangle.

- `tests/test_params.py`
  - Default/JSON/explicit-CLI precedence.
  - Unknown JSON key rejection.
  - Range and cross-field failures.
  - ROI parsing, clipping, and empty-ROI rejection.

- `tests/test_sources.py`
  - Natural filename order.
  - Consistent image sequence dimensions.
  - Unreadable image error at the file's first read.
  - Dimension mismatch error containing the offending path.

- `tests/test_cli.py`
  - `--params` loads a complete tuning output.
  - Explicit CLI values override JSON values.
  - Configuration errors are presented as argparse errors.
  - Video writer open failure is reported before frame processing.

- `tests/test_tuning.py`
  - `thresh` is accepted as a fixed method by the actual argument parser.
  - Unsupported grid keys, including `axis`, are rejected.
  - Unknown, empty, or non-list grid entries are rejected.

Every behavioral production change follows red-green-refactor: add one failing regression test, confirm the expected failure, implement the minimum correction, then run the focused test and full suite.

## Documentation

`README.md` is updated to document:

- New module responsibilities and programmatic import paths.
- The removal of Python imports from `count_cv.py`.
- `--params` and precedence examples.
- `thresh` tuning support.
- Fixed-only grid parameters.
- Consecutive `min_hits` semantics.
- Development dependency installation and `pytest -q`.

Existing operating assumptions and physical limitations remain unchanged.

## Verification

Completion requires all of the following evidence:

1. `pytest -q` passes.
2. All scripts and new modules pass `python3 -m py_compile`.
3. `python3 count_cv.py simp.mp4 --method bgsub --warmup 8` reports `4/4` on the available smoke-test artifact.
4. A small `tune_params.py` grid over `samples.txt` completes with training `SAE=0`.
5. `python3 count_cv.py simp.mp4 --params <generated-json>` exercises JSON loading, and an explicit CLI override is reflected in the resolved behavior or a focused CLI test.

The local videos are smoke-test inputs only; automated tests do not depend on ignored artifacts.

## Risks and Mitigations

- **Import breakage:** Intentional clean cutover. Every repository caller is migrated in the same change; no compatibility aliases remain.
- **Argparse precedence mistakes:** A focused CLI test proves JSON values survive unless an option is explicitly supplied.
- **Tuner cache invalidity:** Unsupported stage-crossing grid keys are rejected instead of silently producing duplicate or incorrect evaluations.
- **Over-validation of existing workflows:** Constraints preserve valid defaults and existing documented ranges; clipping behavior for ROI remains supported.
- **Refactor regression:** Behavior tests are established before moving implementations, and CLI smoke tests verify the actual entry points after the move.
