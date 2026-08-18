# Reproducible Interference Dataset Generator Design

**Date:** 2026-08-18

## Goal

Extend `gen_shapes_video.py` so one deterministic scene can produce both MP4 and a same-resolution JPG frame sequence, with every JPG no larger than a configurable hard limit (60 KiB by default), broader object contours, and parameterized background interference that exposes preprocessing and tracking weaknesses.

## Scope

This change includes:

- Optional JPG frame-sequence output alongside the existing MP4.
- A hard per-frame JPEG size limit with a uniform output resolution.
- Common shapes and complex shapes with multiple protrusions/concavities.
- Reproducible background-interference presets plus independent controls.
- Updated YOLO labels and richer generation metadata.
- Focused generator tests and README usage documentation.

This change does not add a batch benchmark runner, change the counting algorithm, add classification training, add a physics engine, or introduce external image-generation dependencies.

## Compatibility

Existing commands that omit all new generator options retain current behavior:

- One MP4 at the requested width and height.
- The existing `_meta.json` path and core fields.
- Optional YOLO labels.
- Existing shape/noise/blur options, one-stream RNG order, and default seed semantics.

This legacy scene mode keeps the current four-shape uniform distribution (`circle`, `triangle`, `square`, `rectangle`), clean belt rendering, one-stream RNG order, and default seed semantics. `--frames-dir` alone changes only output handling and still uses this legacy scene mode; new shape/background options explicitly activate independent derived RNG streams. Compatibility means byte-identical seeded frames and truth for commands that do not use new scene options, subject to existing codec/platform behavior.

When `--frames-dir` is present, MP4 and JPG outputs use the same rendered frames and the same final uniform dimensions. If the JPEG limit requires downscaling, the MP4 is also written at that final scale so video and frame-directory counting remain directly comparable.

## Architecture

Keep a single script and split its internals into testable units rather than adding a second generator:

```text
argparse/config validation
        |
        v
Scene(seed, config)
  deterministic object/background RNG streams
  spawn/update/render one unscaled frame + truth
        |
        +--> JPEG calibration pass --> uniform output scale
        |
        v
Scene reset with same seed
  final render pass
        +--> MP4 writer
        +--> bounded JPG encoder
        +--> YOLO labels
        +--> object/event metadata
```

`Scene` owns all mutable simulation state: objects, frame index, spawn distance, background phase, truth events, and independent RNG streams. Constructing a new `Scene` from the same validated configuration and seed must reproduce identical uncompressed frames and truth events.

Independent RNG streams are derived deterministically from `numpy.random.SeedSequence(seed)` for:

- Object spawning and shapes.
- Background texture/clutter.
- Illumination/flicker.
- Sensor noise.
- Shadows.
- Camera jitter.

Adding one interference type must not silently change unrelated object geometry for the same seed.

## Output Contract

New CLI options:

```text
--frames-dir DIR
--max-frame-kb 60
--jpeg-quality-min 20
--jpeg-quality-max 95
--jpeg-scale-step 0.85
--overwrite
--min-output-width 160
--min-output-height 90
```

When `--frames-dir` is supplied:

```text
OUTPUT.mp4
OUTPUT_meta.json
DIR/frame_000000.jpg
DIR/frame_000001.jpg
...
```

If `--labels DIR` is also supplied, labels remain:

```text
LABEL_DIR/frame_000000.txt
...
```

All JPG files must satisfy:

```text
file_size <= max_frame_kb * 1024
```

All JPG frames and the MP4 must have one common even width and height. Frame count, ordering, and filenames remain one-to-one across MP4, JPG, and labels.

Each output is written to a staging sibling beside its own final destination, so its final rename and backup operations stay on that destination's filesystem even when MP4, frames, and labels use different mounts. A transaction manifest named `<output>.transaction.json` beside the MP4 records every final, staging, and backup path plus promotion state. A completion marker named `<output>.complete.json` is written only after MP4, metadata, frame directory, and labels have been fully produced and promoted. Existing final outputs are moved to same-filesystem sibling backups in a fixed order, staged siblings are promoted in a fixed order, and failures roll back promoted outputs from their recorded backups and remove the completion marker. A later invocation reads an incomplete transaction manifest, restores every recorded backup, removes stale staging siblings, and only then starts a new run. Readers should treat a run without its completion marker as incomplete. When frame output is requested, a non-empty existing frame/label directory requires explicit `--overwrite`; legacy MP4 overwrite behavior is unchanged.

## JPEG Size Enforcement

Per-frame resizing is forbidden because `FrameSource` requires consistent dimensions and variable dimensions would make frame-directory results incomparable.

Use a deterministic calibration-and-retry loop:

1. Start with output scale `1.0`.
2. Run a calibration scene from frame zero without writing final outputs.
3. For each unscaled rendered frame, resize to the current global scale and attempt JPEG encoding at `jpeg_quality_min`.
4. If the frame exceeds the byte limit, multiply the global scale by `jpeg_scale_step`, round dimensions down to positive even values, and retry that frame until it fits.
5. Continue calibration with the smaller global scale. Frames that fit at a larger scale are conservatively assumed to fit at the smaller candidate scale.
6. Reset the scene from the same seed and perform an output pass at the final scale.
7. For every output frame, binary-search the highest integer JPEG quality in `[jpeg_quality_min, jpeg_quality_max]` whose encoded bytes meet the limit.
8. If any output-pass frame cannot fit at `jpeg_quality_min`, discard staging outputs, lower the global scale, and restart the complete calibration plus output pass. Repeat this loop until the output pass succeeds.
9. If lowering scale does not change the rounded even dimensions, or the next scale would violate minimum output dimensions, raise `RuntimeError` naming the frame, byte limit, minimum quality, and minimum dimensions. This is the explicit termination condition.

The MP4 writer is opened only after final dimensions are known. Its `isOpened()` result is checked.

The metadata records final scale, dimensions, quality range used, and byte-size min/mean/p95/max. Tests verify the actual filesystem byte sizes, not only encoder buffer lengths.

## Object Families

Supported names:

### Basic

- `circle`
- `ellipse`
- `triangle`
- `square`
- `rectangle`
- `pentagon`
- `hexagon`

### Protruding and concave

- `star`
- `gear`
- `cross`
- `t_shape`
- `l_shape`
- `bottle`
- `dumbbell`
- `peanut`
- `lobed`

New CLI options:

```text
--shape-profile legacy|basic|protruding|mixed
--shape-weights JSON
--protrusions-min 1
--protrusions-max 5
--concavity 0.35
--rotation-speed-min 0
--rotation-speed-max 0
--aspect-min 1.0
--aspect-max 2.4
```

The default is `legacy` when no new options are supplied. `legacy` uses exactly the existing four shapes with equal weights. `basic` uses equal weights over the seven basic shapes; `protruding` uses equal weights over the nine complex shapes; `mixed` uses equal total weight for the basic and complex families. `--shape-weights` overrides profile weights using a JSON object whose keys are supported shape names and whose values are finite non-negative numbers with at least one positive value.

Shape generation rules:

- All outputs are simple closed polygons or sampled simple contours.
- Base contours are centered at the origin and normalized to unit maximum radius.
- `star`, `gear`, and `lobed` use ordered polar vertices with alternating/deterministic radial structure.
- `cross`, `t_shape`, and `l_shape` use fixed ordered orthogonal templates.
- `bottle`, `dumbbell`, and `peanut` use ordered contour samples around a symmetric profile; no Boolean geometry dependency is added.
- Existing radial/tangential deformation, wobble, rotation, and size scaling apply after base-contour construction.
- After every deformation and per-frame wobble transform, validate positive area and test every non-adjacent edge pair for intersection using an orientation test with an epsilon. If invalid, deterministically halve the effective deformation and wobble contribution and retry up to 8 times; if still invalid, use the valid base contour at the same center/rotation. Never emit an invalid contour; if even the base contour is invalid, raise `RuntimeError` with seed, object ID, and frame.
- Rotation speed is per-object and deterministic.


YOLO remains a single class (`0`) because this is a counting dataset. Object shape names and IDs are stored in metadata instead of changing the task to classification.

## Background Interference

New profile option:

```text
--background-profile clean|moving-texture|illumination|camera-jitter|shadow|clutter|mixed
```

Profiles provide defaults; explicitly supplied independent controls override profile values.

Independent controls:

```text
--texture-type none|stripe|grid|checker|noise
--texture-contrast FLOAT
--texture-speed FLOAT
--texture-scale FLOAT
--gradient-strength FLOAT
--flicker-amplitude FLOAT
--flicker-frequency FLOAT
--brightness-drift FLOAT
--camera-jitter-px FLOAT
--camera-jitter-frequency FLOAT
--shadow-count INT
--shadow-opacity FLOAT
--shadow-speed FLOAT
--clutter-count INT
--clutter-saturation FLOAT
--sensor-noise FLOAT
```

Rendering order:

1. Base belt color and deterministic gradient.
2. Static or moving texture.
3. Static clutter that is not part of truth.
4. Objects.
5. Moving soft shadows that are not part of truth.
6. Global illumination drift and flicker.
7. Whole-frame camera jitter transform.
8. Sensor noise.
9. Existing motion blur.

Background primitives use only OpenCV/NumPy. Soft shadows may use a small blurred alpha mask. Camera jitter uses an affine translation with deterministic sinusoidal and seeded components; border fill uses the current background color rather than black.

`--noise` remains accepted as the existing sensor-noise option. `--sensor-noise` is an explicit alias, and specifying both is rejected rather than assigned ambiguous precedence.

## Truth and Labels

Truth is computed in final camera coordinates after camera jitter because the counting line is fixed in the output image.

Each object has:

```text
id
shape
color
size
aspect
spawn_frame
crossing_frame or null
```

An object counts once when its jitter-transformed center first crosses the fixed output center line in the configured direction. Background clutter and shadows never contribute truth.

Normalized YOLO boxes are computed after camera jitter and clipped to the final frame. Fully off-screen objects are omitted from that frame's label file. Uniform output scaling does not change normalized labels.

The metadata retains existing fields and adds:

```json
{
  "seed": 0,
  "shape_profile": "mixed",
  "shape_counts": {"circle": 4, "gear": 3},
  "background_profile": "mixed",
  "interference": {},
  "objects": [],
  "frames_dir": "frames_case",
  "jpeg_limit_bytes": 61440,
  "jpeg_frame_scale": 0.5,
  "jpeg_quality_min_used": 42,
  "jpeg_quality_max_used": 91,
  "jpeg_bytes_min": 20110,
  "jpeg_bytes_mean": 35500.0,
  "jpeg_bytes_p95": 57700.0,
  "jpeg_bytes_max": 61234,
  "output_resolution": [640, 360]
}
```
## Licensed Twemoji Outline Assets

Emojipedia is used only as a category reference. Its About page states that displayed emoji images belong to their respective creators, so the generator must not scrape or redistribute those vendor images.

The concrete image source is Twemoji v17.0.3 from the official `jdecked/twemoji` repository. Twemoji graphics are licensed under CC BY 4.0. The repository commits:

- A curated manifest of outline-distinct Unicode objects covering tools, household items, office objects, electronics, music, containers, keys/locks, and equipment.
- Processed transparent silhouette PNGs only, not the downloaded source PNGs.
- The Twemoji graphics license and attribution notice.
- Per-asset Unicode code point, CLDR-style name, upstream URL, Twemoji version, source SHA256, processed SHA256, and crop dimensions.

Add a reproducible maintenance script `prepare_twemoji_objects.py`. It downloads only manifest-listed PNGs from version-pinned URLs, verifies HTTP success and source hashes when already recorded, and reads with `cv2.IMREAD_UNCHANGED`. Assets without a usable alpha channel are rejected.

Processing is deterministic:

1. Extract the source alpha channel; do not infer foreground from white, black, or colored rectangular backgrounds.
2. Crop to the nonzero-alpha bounding box, preserving transparent holes inside the object.
3. Threshold alpha at a fixed manifest-wide level to form the subject mask.
4. Keep every connected component above a documented relative-area floor so detached but meaningful parts survive; remove only tiny antialiasing specks.
5. Normalize into a square transparent canvas with fixed padding while preserving aspect ratio.
6. Store the result as a single-channel transparent silhouette PNG with foreground alpha and zero RGB. The generator chooses the final object color/gray value.
7. Validate positive foreground area and record both source and processed hashes.

The runtime generator never downloads. It loads committed processed assets through a new shape profile:

```text
--shape-profile twemoji|mixed-assets
--twemoji-manifest PATH
```

`twemoji` samples only processed silhouettes. `mixed-assets` assigns equal total probability to procedural contours and Twemoji silhouettes. `--shape-weights` may refer to manifest asset names as well as procedural names.

Runtime compositing uses the silhouette alpha only:

- Tight nonzero-alpha bounds define the source subject, not the PNG rectangle.
- Resize and rotate a premultiplied alpha mask with antialiased interpolation.
- Choose color using the existing gray/color object rules.
- Composite only through alpha; transparent pixels never overwrite the rendered belt/background.
- Compute bounding boxes, centroid, labels, and truth position from the transformed nonzero alpha support.
- Preserve internal transparent holes such as key rings and scissors handles.

Tests cover pinned download URLs with a local HTTP fixture, alpha-required rejection, deterministic crop/hash output, preservation of a detached meaningful component and an internal hole, absence of rectangular-background contamination after compositing, manifest/license completeness, and runtime reproducibility for a fixed seed.

Metadata serialization is deterministic: object/event lists are ID/frame ordered and dictionaries use stable construction order. Output paths may differ between runs, so reproducibility tests compare rendered files and normalized metadata fields rather than requiring path fields to match.


## Validation

Reject before rendering:

- Non-positive duration, FPS, dimensions, count, or size.
- Invalid min/max relationships.
- `max_frame_kb <= 0`.
- JPEG qualities outside 1–100 or min greater than max.
- Invalid min/max relationships.
- Default-less new options that silently change legacy shape/RNG behavior (new shape/background options must select an explicit non-legacy mode).
- Minimum output dimensions larger than requested dimensions.
- Negative counts, noise, contrast, jitter, shadow, clutter, or protrusion values.
- Opacity outside 0–1.
- Invalid shape-weight JSON or unknown shape names.
- An existing non-empty output/frame/label directory unless the current explicit overwrite behavior permits safe replacement. The implementation must not silently mix files from different runs.

## Tests

Add focused tests without checking large binary artifacts into Git:

- Same seed and configuration produce identical JPG hashes, labels, truth events, and normalized metadata.
- Different seeds change at least one rendered frame and object record.
- Every JPG is at or below the configured byte limit on a noisy `mixed` profile.
- Every JPG and decoded MP4 frame has the same final dimensions and frame count.
- Forced small limits select a scale below 1.0 while preserving one resolution across all frames.
- Impossible limit/minimum-dimension combinations fail and leave no completed output directory.
- Every supported shape returns a finite, non-self-intersecting normalized contour with positive area.
- Protruding profile produces at least one concave/multi-lobed contour for a fixed seed.
- Background profiles are deterministic and measurably differ from `clean` for a fixed scene.
- Camera jitter updates boxes and crossing truth in final camera coordinates.
- Shadows and clutter appear in pixels but never in labels or count truth.
- Existing default command without `--frames-dir` preserves requested MP4 dimensions and core metadata fields.

Tests use short, low-resolution scenes and temporary directories. The real smoke test generates a short `mixed` scene with `--frames-dir`, confirms every JPG is `<= 60 * 1024` bytes, and runs `count_cv.py` on the frame directory when the hardened counter branch is available.

## Documentation

Update the generator section of `README.md` with:

- New shape profiles and examples.
- Background profile and independent override examples.
- MP4 + JPG command example.
- Exact 60 KiB guarantee and possible uniform downscaling.
- Reproducibility via `--seed`.
- Explanation that complex/interference cases are intentionally adversarial and may produce count errors.

## Risks and Mitigations

- **Two-pass cost:** Calibration renders the scene once; output normally renders once more. Full retries happen only if encoder-size assumptions fail. No frames are retained in memory.
- **RNG drift between passes:** All state is owned by a newly constructed `Scene` and independent derived RNG streams; reproducibility tests compare output hashes.
- **Label/truth mismatch under jitter:** Boxes and crossing events are computed after the same affine transform applied to pixels.
- **Variable JPG dimensions:** Forbidden; one global even resolution is selected before final output.
- **Partial outputs:** Staging paths are promoted only after successful completion.
- **Scope growth:** No benchmark matrix runner or counting changes are included in this phase.
