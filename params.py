"""Parameter defaults, loading, validation, and stage ownership."""

import json
import math
from collections.abc import Sequence


DEFAULT_PARAMS = {
    # 检测
    "method": "auto", "sat_thresh": 60,
    "thresh_lo": 50, "thresh_hi": 205,
    "min_area": 300, "max_area": 0, "max_aspect": 0.0,
    "min_area_frac": 0.0, "max_area_frac": 0.0,
    "morph_kernel": 7, "morph_iter": 2,
    "bg_history": 200, "bg_var": 40.0,
    "ref_thresh": 25, "bg_ref": None, "ref_alpha": 0.0,
    "split_area": 0, "unit_area": 0, "merge_dist": 0.0,
    "roi": None, "scale": 1.0,
    # 跟踪/计数
    "max_dist": 140.0, "track_ttl": 5,
    "min_hits": 1, "min_speed": 0.0,
    "line": 0.5, "line_band": 0.0,
    "axis": "x", "flow": "both", "warmup": 8,
}

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

_INTEGER_FIELDS = {
    "sat_thresh", "thresh_lo", "thresh_hi", "min_area", "max_area",
    "morph_kernel", "morph_iter", "bg_history", "ref_thresh",
    "split_area", "unit_area", "track_ttl", "min_hits", "warmup",
}
_NUMERIC_FIELDS = {
    "max_aspect", "min_area_frac", "max_area_frac", "bg_var",
    "ref_alpha", "merge_dist", "scale", "max_dist", "min_speed",
    "line", "line_band",
}
_NONNEGATIVE_FIELDS = {
    "min_area", "max_area", "max_aspect", "morph_kernel", "morph_iter",
    "bg_history", "bg_var", "split_area", "unit_area", "merge_dist",
    "track_ttl", "min_speed", "warmup",
}


def load_params(path):
    """Load a partial parameter object from a JSON file."""
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
    """Overlay defaults, file values, and explicit values in precedence order."""
    merged = dict(DEFAULT_PARAMS)
    merged.update(file_params or {})
    merged.update(cli_params or {})
    return validate_params(merged)


def _roi_values(roi):
    message = "roi must contain four integers: x0,y0,x1,y1"
    if isinstance(roi, str):
        parts = roi.split(",")
        if len(parts) != 4:
            raise ValueError(message)
        try:
            return tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(message) from exc

    if not isinstance(roi, Sequence) or len(roi) != 4:
        raise ValueError(message)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
        raise ValueError(message)
    return tuple(roi)


def parse_roi(roi, w, h):
    """Parse and clip an ROI to a frame, rejecting empty clipped regions."""
    if roi is None:
        return None
    x0, y0, x1, y1 = _roi_values(roi)
    clipped = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"roi is empty after clipping to {w}x{h}: {roi}")
    return clipped


def _require_integer(params, name):
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_numeric(params, name):
    value = params[name]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number")


def validate_params(params, width=None, height=None):
    """Validate a complete parameter dictionary and return it unchanged."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    expected = set(DEFAULT_PARAMS)
    actual = set(params)
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing parameter(s): {', '.join(missing)}")

    for name in _INTEGER_FIELDS:
        _require_integer(params, name)
    for name in _NUMERIC_FIELDS:
        _require_numeric(params, name)

    if params["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: {params['method']}")
    if params["axis"] not in AXES:
        raise ValueError(f"axis must be one of {AXES}: {params['axis']}")
    if params["flow"] not in FLOWS:
        raise ValueError(f"flow must be one of {FLOWS}: {params['flow']}")
    if params["bg_ref"] is not None and not isinstance(params["bg_ref"], str):
        raise ValueError("bg_ref must be a string or null")

    if params["scale"] <= 0:
        raise ValueError("scale must be greater than zero")

    for name in ("sat_thresh", "thresh_lo", "thresh_hi", "ref_thresh"):
        if not 0 <= params[name] <= 255:
            raise ValueError(f"{name} must be within 0..255")
    if params["thresh_lo"] > params["thresh_hi"]:
        raise ValueError("thresh_lo must not exceed thresh_hi")

    for name in _NONNEGATIVE_FIELDS:
        if params[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    if params["min_hits"] < 1:
        raise ValueError("min_hits must be at least 1")
    if params["max_dist"] <= 0:
        raise ValueError("max_dist must be greater than zero")

    if not 0 <= params["line"] <= 1:
        raise ValueError("line must be within 0..1")
    if not 0 <= params["line_band"] <= 0.5:
        raise ValueError("line_band must be within 0..0.5")
    for name in ("min_area_frac", "max_area_frac", "ref_alpha"):
        if not 0 <= params[name] <= 1:
            raise ValueError(f"{name} must be within 0..1")

    if (params["min_area"] > 0 and params["max_area"] > 0
            and params["max_area"] < params["min_area"]):
        raise ValueError("max_area must not be below active min_area")
    if (params["min_area_frac"] > 0 and params["max_area_frac"] > 0
            and params["max_area_frac"] < params["min_area_frac"]):
        raise ValueError("max_area_frac must not be below active min_area_frac")

    if params["roi"] is not None:
        _roi_values(params["roi"])
        if width is not None and height is not None:
            parse_roi(params["roi"], width, height)

    return params
