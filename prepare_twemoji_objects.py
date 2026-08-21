#!/usr/bin/env python3
"""Download pinned Twemoji PNGs and prepare transparent silhouette assets.

The input manifest is deliberately small and is the only source of download URLs.
Runtime generation consumes the processed PNGs; this utility is the maintenance step
that may access the network.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def _sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(
    url: str,
    destination: PathLike,
    expected_sha256: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """Download one source PNG, verify its optional hash, and return its hash."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is not None and status != 200:
                raise ValueError("source download returned HTTP status %s" % status)
            with destination_path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise

    actual = _sha256(destination_path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        destination_path.unlink(missing_ok=True)
        raise ValueError("source SHA256 mismatch: expected %s, got %s" % (expected_sha256, actual))
    return actual


def process_rgba(
    source_path: PathLike,
    output_path: PathLike,
    area_floor: float = 0.002,
) -> Dict[str, Any]:
    """Convert one connected RGBA object to a padded square alpha silhouette.

    Isolated antialiasing specks are ignored, but an image containing two or
    more meaningful foreground components is rejected.  This prevents assets
    such as separated tool sets or detached accessories from entering the
    generator.
    """
    if area_floor < 0:
        raise ValueError("area_floor must be non-negative")
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("source image must have an alpha channel")

    alpha = source[:, :, 3]
    support = (alpha >= 1).astype(np.uint8)  # fixed 1/255 support threshold
    if not np.any(support):
        raise ValueError("source image has no usable alpha")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros_like(support)
    component_floor = max(0.002, area_floor) * largest
    minimum_component_area = 2
    kept_components = 0
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= minimum_component_area and area >= component_floor:
            keep[labels == component] = 1
            kept_components += 1
    if not np.any(keep):
        raise ValueError("source image has no component above area floor")
    if kept_components != 1:
        component_areas = sorted(
            (
                int(stats[component, cv2.CC_STAT_AREA])
                for component in range(1, count)
                if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component_area
                and int(stats[component, cv2.CC_STAT_AREA]) >= component_floor
            ),
            reverse=True,
        )
        raise ValueError(
            "source foreground must be one connected component; found %d "
            "meaningful components with areas %s"
            % (kept_components, component_areas)
        )

    ys, xs = np.where(keep > 0)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    cropped_alpha = alpha[top:bottom, left:right].copy()
    cropped_keep = keep[top:bottom, left:right]
    cropped_alpha[cropped_keep == 0] = 0

    # Fixed padding makes outputs comparable while retaining every hole and part.
    padding = 4
    side = max(cropped_alpha.shape) + padding * 2
    result = np.zeros((side, side, 4), dtype=np.uint8)
    y0 = padding + (side - padding * 2 - cropped_alpha.shape[0]) // 2
    x0 = padding + (side - padding * 2 - cropped_alpha.shape[1]) // 2
    result[y0:y0 + cropped_alpha.shape[0], x0:x0 + cropped_alpha.shape[1], 3] = cropped_alpha

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError("failed to write processed image: %s" % output)
    return {
        "processed_sha256": _sha256(output),
        "width": int(side),
        "height": int(side),
        "foreground_area": int(np.count_nonzero(result[:, :, 3])),
        "components": kept_components,
        "connectivity": 8,
        "padding": padding,
    }


def validate_manifest(data: Dict[str, Any]) -> bool:
    """Validate manifest structure and metadata without requiring hashes yet."""
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    if data.get("twemoji_version") != "17.0.3":
        raise ValueError("manifest must pin Twemoji version 17.0.3")
    license_data = data.get("license")
    if not isinstance(license_data, dict) or not license_data.get("name") or not license_data.get("url"):
        raise ValueError("manifest requires license name and URL")
    if license_data["name"] != "CC BY 4.0":
        raise ValueError("manifest must use CC BY 4.0")
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("manifest requires a non-empty assets list")

    required = ("asset", "unicode", "name", "source_url", "source_sha256", "processed_sha256", "license", "twemoji_version")
    seen = set()
    for item in assets:
        if not isinstance(item, dict) or any(not item.get(key) and key not in ("source_sha256", "processed_sha256") for key in required):
            raise ValueError("each asset requires complete metadata")
        asset = item["asset"]
        if asset in seen:
            raise ValueError("duplicate asset name: %s" % asset)
        seen.add(asset)
        if item["license"] != "CC BY 4.0" or item["twemoji_version"] != "17.0.3":
            raise ValueError("asset license/version metadata is invalid")
        if "emojipedia" in item["source_url"].lower():
            raise ValueError("Emojipedia URLs are not permitted")
        source_url = item["source_url"]
        official_url = "https://raw.githubusercontent.com/jdecked/twemoji/v17.0.3/"
        # Network sources must be pinned to the official Twemoji release.  Local
        # ``file:`` URLs remain supported for offline fixtures/tests.
        if source_url.startswith(("http://", "https://")) and not source_url.startswith(official_url):
            raise ValueError("HTTP(S) source URL must be pinned to official Twemoji v17.0.3")
        if not source_url.startswith(("file:", "http:", "https:")):
            raise ValueError("source URL must use file or HTTP(S) scheme")
        for field in ("source_sha256", "processed_sha256"):
            value = item[field]
            if value and (not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value)):
                raise ValueError("%s must be a SHA256 hex digest" % field)
    return True


def prepare_manifest(
    manifest_path: PathLike,
    output_dir: PathLike,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Prepare all manifest assets and rewrite deterministic source/processed hashes."""
    manifest_file = Path(manifest_path)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    validate_manifest(data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="twemoji-source-") as temporary:
        temporary_dir = Path(temporary)
        for item in data["assets"]:
            source_file = temporary_dir / (item["asset"] + ".png")
            source_hash = download_source(item["source_url"], source_file, item.get("source_sha256") or None, timeout=timeout)
            item["source_sha256"] = source_hash
            processed_file = output / (item["asset"] + ".png")
            details = process_rgba(source_file, processed_file)
            item.update(details)
            item["processed_sha256"] = details["processed_sha256"]

    # Preserve curated asset order, but make all object key ordering deterministic.
    manifest_file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to the Twemoji manifest JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for processed silhouette PNGs")
    parser.add_argument("--timeout", type=int, default=30, help="network timeout in seconds")
    args = parser.parse_args()
    prepare_manifest(args.manifest, args.out_dir, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
