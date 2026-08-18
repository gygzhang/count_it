# Twemoji Object Silhouette Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superartes:subagent-driven-development (recommended) or superartes:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible, licensed Twemoji object silhouettes to the synthetic generator without downloading or redistributing Emojipedia/vendor artwork and without contaminating rendered backgrounds.

**Architecture:** Add a pinned asset manifest and a preparation utility that converts Twemoji RGBA PNGs into transparent single-color silhouette PNGs. Extend `gen_shapes_video.py` to load local processed assets, composite them through Alpha only, compute transformed truth/boxes from Alpha support, and preserve the existing procedural generator behavior when the new asset profile is not selected.

**Tech Stack:** Python 3, OpenCV, NumPy, urllib, hashlib, JSON, pytest

**Spec:** `docs/specs/2026-08-18-reproducible-interference-generator-design.md`

---

## File Map

- Create `twemoji_objects_manifest.json`: curated Unicode/name/source URL/hash/license metadata.
- Create `THIRD_PARTY_TWEMOJI_LICENSE.txt`: Twemoji CC BY 4.0 text/attribution.
- Create `prepare_twemoji_objects.py`: pinned download, SHA256 verification, alpha crop, silhouette conversion.
- Create `assets/twemoji_objects/*.png`: processed transparent silhouettes only.
- Modify `gen_shapes_video.py`: asset loading, `twemoji`/`mixed-assets` profiles, Alpha compositing, transformed truth/labels, CLI validation.
- Create `tests/test_twemoji_assets.py`: preparation and manifest tests using local fixtures; no network dependency.
- Create `tests/test_twemoji_generator.py`: runtime compositing, transparency, holes, labels, reproducibility.
- Modify `README.md`: licensed asset setup and generator usage.

### Task 1: Define manifest, license, and deterministic asset preparation

**Files:**
- Create: `twemoji_objects_manifest.json`
- Create: `THIRD_PARTY_TWEMOJI_LICENSE.txt`
- Create: `prepare_twemoji_objects.py`
- Create: `tests/test_twemoji_assets.py`

- [ ] **Step 1: Add failing preparation tests**

Test with local temporary source PNGs and a local HTTP fixture or monkeypatched downloader:

```python
def test_rejects_source_without_alpha(tmp_path): ...
def test_crops_transparent_margin_and_preserves_hole(tmp_path): ...
def test_preserves_detached_component_above_area_floor(tmp_path): ...
def test_rejects_bad_source_hash(tmp_path): ...
def test_manifest_requires_license_version_url_and_hash(): ...
def test_processing_is_deterministic(tmp_path): ...
```

The fixture must contain an RGBA object with transparent margins, one internal transparent hole, and one detached meaningful component. Assert output alpha, dimensions, hole, component, and SHA256 are stable.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_twemoji_assets.py
```

Expected: collection fails because `prepare_twemoji_objects.py` and the manifest do not exist.

- [ ] **Step 3: Create the pinned manifest and license**

Use Twemoji v17.0.3 URLs under the official `jdecked/twemoji` repository. Include a curated outline-distinct subset covering tools, household, office, electronics, music, containers, locks/keys, and equipment. Each record contains:

```json
{
  "asset": "hammer",
  "unicode": "1f528",
  "name": "hammer",
  "source_url": "https://raw.githubusercontent.com/jdecked/twemoji/v17.0.3/assets/72x72/1f528.png",
  "source_sha256": "",
  "processed_sha256": "",
  "license": "CC BY 4.0",
  "twemoji_version": "17.0.3"
}
```

Do not add Emojipedia image URLs. Empty hashes are allowed only in the initial maintainer manifest before preparation; the preparation script must write verified hashes back deterministically.

- [ ] **Step 4: Implement alpha-only preparation**

Implement functions with stable signatures:

```python
def download_source(url, destination, expected_sha256=None): ...
def process_rgba(source_path, output_path, area_floor=0.002): ...
def prepare_manifest(manifest_path, output_dir, timeout=30): ...
def validate_manifest(data): ...
```

Rules:

- Download only manifest-listed URLs.
- Read using `cv2.IMREAD_UNCHANGED`; reject missing/invalid Alpha.
- Crop to nonzero alpha bounds.
- Threshold Alpha at fixed `1/255` support.
- Preserve connected components above `area_floor * largest_component_area`; remove only smaller antialias specks.
- Preserve internal holes.
- Normalize to a square transparent canvas with fixed padding.
- Write zero RGB with Alpha subject mask.
- Record source/processed hashes and dimensions.
- Never use white/black/color thresholding to infer foreground.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run `pytest -q tests/test_twemoji_assets.py`; expected all preparation tests pass. Run `python3 -m py_compile prepare_twemoji_objects.py`.

- [ ] **Step 6: Commit the asset-preparation checkpoint**

Use the commit-message skill and commit the manifest, license, preparation utility, and tests. Suggested subject: `Add licensed Twemoji silhouette preparation`.

### Task 2: Integrate silhouettes into the generator

**Files:**
- Modify: `gen_shapes_video.py`
- Create: `tests/test_twemoji_generator.py`

- [ ] **Step 1: Add failing runtime tests**

Create a tiny processed silhouette fixture and test:

```python
def test_alpha_composite_does_not_paint_png_rectangle(): ...
def test_internal_transparent_hole_reaches_background(): ...
def test_alpha_support_defines_bbox_and_centroid(): ...
def test_twemoji_profile_is_reproducible_for_seed(tmp_path): ...
def test_mixed_assets_can_select_procedural_or_twemoji(tmp_path): ...
```

Assert that only Alpha support changes the background, labels use the transformed support bounds, and two identical seeds produce identical frame bytes and metadata events.

- [ ] **Step 2: Run focused tests and verify RED**

Run `pytest -q tests/test_twemoji_generator.py`; expected failure because generator asset profile and Alpha compositor are absent.

- [ ] **Step 3: Add asset configuration and validation**

Add CLI options:

```text
--shape-profile legacy|basic|protruding|mixed|twemoji|mixed-assets
--twemoji-manifest PATH
```

Default remains `legacy` for existing commands. The `twemoji` profile samples only processed manifest assets; `mixed-assets` assigns equal total probability to procedural and asset families. Validate manifest path, nonempty entries, supported asset names, and positive weights before rendering.

- [ ] **Step 4: Implement local asset loading**

Add a loader that reads processed PNGs with `cv2.IMREAD_UNCHANGED`, requires Alpha, crops/normalizes any remaining transparent margin, and returns an immutable asset record containing name, alpha mask, and manifest metadata. Runtime generation must never download.

- [ ] **Step 5: Implement Alpha-safe object rendering**

For an asset object:

1. Resize and rotate the Alpha mask with antialiased interpolation.
2. Choose a final gray/color foreground using existing object color rules.
3. Composite only Alpha support onto the frame; transparent pixels leave the belt unchanged.
4. Derive transformed bbox and centroid from nonzero Alpha support.
5. Preserve internal holes and detached meaningful components.
6. Apply existing object motion, wobble, and rotation semantics.

Use premultiplied Alpha or equivalent float blending to avoid black/white halos.

- [ ] **Step 6: Integrate truth and labels**

Keep YOLO class `0`. Compute normalized boxes after all object transforms and clip to final frame. Compute crossing truth from transformed object center. Store asset name/Unicode/object ID in metadata; assets and procedural objects share the same ID namespace.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/test_twemoji_generator.py tests/test_twemoji_assets.py
python3 -m py_compile gen_shapes_video.py prepare_twemoji_objects.py
```

Expected: all asset and runtime tests pass.

- [ ] **Step 8: Commit the generator integration checkpoint**

Use the commit-message skill. Suggested subject: `Integrate Twemoji object silhouettes`.

### Task 3: Documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document licensed preparation and runtime use**

Add commands:

```bash
python3 prepare_twemoji_objects.py twemoji_objects_manifest.json \
    --out-dir assets/twemoji_objects
python3 gen_shapes_video.py -o objects.mp4 \
    --shape-profile twemoji \
    --twemoji-manifest twemoji_objects_manifest.json \
    --frames-dir frames_objects \
    --max-frame-kb 60 \
    --seed 7
```

State that Emojipedia supplies category reference only; Twemoji v17.0.3 is the licensed image source; processed resources are silhouettes; Alpha prevents rectangular backgrounds from entering generated frames; and the license/attribution file must travel with redistributed assets/data.

- [ ] **Step 2: Run final verification**

Run:

```bash
pytest -q
python3 -m py_compile gen_shapes_video.py prepare_twemoji_objects.py
python3 prepare_twemoji_objects.py --help
python3 gen_shapes_video.py --help
```

Generate a short local fixture scene using processed assets and verify:

- Every output frame has a single common resolution.
- Every JPG is `<= 60 * 1024` bytes when frame output is enabled.
- Metadata contains asset name/Unicode and source license/version.
- Background pixels remain unchanged outside Alpha support.

- [ ] **Step 3: Commit documentation and verified integration**

Use the commit-message skill. Suggested subject: `Document Twemoji silhouette generation`.

## Completion Criteria

- No runtime network access is required.
- No Emojipedia/vendor image is downloaded or committed.
- Only processed transparent silhouette PNGs are used at runtime.
- Alpha holes and detached meaningful components are preserved.
- Legacy generator behavior remains unchanged when no new scene/profile option is used.
- Twemoji output is reproducible for a fixed seed.
- All tests pass and the 60 KiB JPG contract is verified from filesystem bytes.
