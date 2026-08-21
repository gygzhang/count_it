import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from prepare_twemoji_objects import (
    download_source,
    prepare_manifest,
    process_rgba,
    validate_manifest,
)


def _write_rgba(path: Path, with_alpha=True) -> None:
    image = np.zeros((32, 40, 4 if with_alpha else 3), dtype=np.uint8)
    if with_alpha:
        alpha = image[:, :, 3]
        alpha[7:25, 8:28] = 255
        alpha[12:19, 14:21] = 0  # internal transparent hole
        alpha[13:17, 30:35] = 255  # detached meaningful component
        alpha[1, 1] = 1  # tiny antialias speck, removed by area floor
    else:
        image[:, :, :3] = 255
    assert cv2.imwrite(str(path), image)


def test_rejects_source_without_alpha(tmp_path):
    source = tmp_path / "rgb.png"
    output = tmp_path / "out.png"
    _write_rgba(source, with_alpha=False)

    with pytest.raises(ValueError, match="alpha"):
        process_rgba(source, output)


def test_crops_transparent_margin_and_preserves_hole(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    image = np.zeros((32, 40, 4), dtype=np.uint8)
    image[7:25, 8:28, 3] = 255
    image[12:19, 14:21, 3] = 0
    assert cv2.imwrite(str(source), image)

    result = process_rgba(source, output)
    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image.shape[0] == image.shape[1]
    assert image.shape[2] == 4
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 0)
    assert ys.min() > 0 and xs.min() > 0
    assert ys.max() < alpha.shape[0] - 1 and xs.max() < alpha.shape[1] - 1
    # The hole survives as transparent alpha, and foreground RGB is zero.
    assert np.any(alpha == 0)
    assert np.all(image[:, :, :3][alpha > 0] == 0)
    assert result["width"] == result["height"] == image.shape[0]


def test_rejects_detached_meaningful_component(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    _write_rgba(source)

    with pytest.raises(ValueError, match="one connected component"):
        process_rgba(source, output, area_floor=0.002)

def test_removes_isolated_one_pixel_speck(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    image = np.zeros((32, 40, 4), dtype=np.uint8)
    image[7:25, 8:28, 3] = 255
    image[1, 1, 3] = 1
    assert cv2.imwrite(str(source), image)

    process_rgba(source, output, area_floor=0.0)
    alpha = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)[:, :, 3]
    count, _, stats, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    assert count - 1 == 1
    assert 1 not in areas
    assert areas[0] == 360


def test_rejects_bad_source_hash(tmp_path):
    source = tmp_path / "source.png"
    _write_rgba(source)
    destination = tmp_path / "copy.png"

    with pytest.raises(ValueError, match="SHA256"):
        download_source(source.as_uri(), destination, expected_sha256="0" * 64)


def test_manifest_requires_license_version_url_and_hash():
    valid = {
        "twemoji_version": "17.0.3",
        "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "assets": [{
            "asset": "hammer", "unicode": "1f528", "name": "hammer",
            "source_url": "https://raw.githubusercontent.com/jdecked/twemoji/v17.0.3/assets/72x72/1f528.png",
            "source_sha256": "", "processed_sha256": "", "license": "CC BY 4.0",
            "twemoji_version": "17.0.3",
        }],
    }
    assert validate_manifest(valid) is True
    for field in ("license", "twemoji_version", "assets"):
        invalid = json.loads(json.dumps(valid))
        invalid.pop(field)
        with pytest.raises(ValueError):
            validate_manifest(invalid)


def test_processing_is_deterministic(tmp_path):
    source = tmp_path / "source.png"
    image = np.zeros((32, 40, 4), dtype=np.uint8)
    image[7:25, 8:28, 3] = 255
    image[12:19, 14:21, 3] = 0
    assert cv2.imwrite(str(source), image)
    first, second = tmp_path / "one.png", tmp_path / "two.png"

    result_one = process_rgba(source, first)
    result_two = process_rgba(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert result_one["processed_sha256"] == result_two["processed_sha256"]
    assert result_one["processed_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()


def test_prepare_manifest_uses_downloaded_source_and_records_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    image = np.zeros((32, 40, 4), dtype=np.uint8)
    image[7:25, 8:28, 3] = 255
    assert cv2.imwrite(str(source), image)
    manifest = tmp_path / "manifest.json"
    data = {
        "twemoji_version": "17.0.3",
        "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "assets": [{
            "asset": "hammer", "unicode": "1f528", "name": "hammer",
            "source_url": source.as_uri(), "source_sha256": "", "processed_sha256": "",
            "license": "CC BY 4.0", "twemoji_version": "17.0.3",
        }],
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")
    def copy_source(url, destination, expected_sha256=None, timeout=30):
        Path(destination).write_bytes(source.read_bytes())
        return hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr("prepare_twemoji_objects.download_source", copy_source)

    output_dir = tmp_path / "assets"
    result = prepare_manifest(manifest, output_dir)
    assert result["assets"][0]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["assets"][0]["processed_sha256"]
    assert result["assets"][0]["components"] == 1
    assert result["assets"][0]["connectivity"] == 8
    assert (output_dir / "hammer.png").exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["assets"][0]["processed_sha256"]
