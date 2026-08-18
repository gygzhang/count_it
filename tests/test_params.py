import json

import pytest

from params import DEFAULT_PARAMS, load_params, merge_params, parse_roi, validate_params


def complete(**changes):
    return {**DEFAULT_PARAMS, **changes}


def test_explicit_values_override_file_and_file_overrides_defaults(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps({"max_dist": 90, "min_hits": 3}), encoding="utf-8"
    )

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


@pytest.mark.parametrize("document", ["[]", "null", "1", '"params"'])
def test_load_params_rejects_non_object_json(tmp_path, document):
    path = tmp_path / "params.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="object.*params.json"):
        load_params(path)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("method", "unknown"),
        ("axis", "z"),
        ("flow", "forward"),
    ],
)
def test_validate_params_rejects_invalid_enums(name, value):
    with pytest.raises(ValueError, match=name):
        validate_params(complete(**{name: value}))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sat_thresh", 1.5),
        ("morph_iter", 1.5),
        ("min_hits", 1.5),
        ("scale", "1.0"),
        ("max_dist", None),
        ("line", True),
        ("warmup", False),
    ],
)
def test_validate_params_rejects_invalid_numeric_types(name, value):
    with pytest.raises(ValueError, match=name):
        validate_params(complete(**{name: value}))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("scale", 0),
        ("scale", -0.1),
        ("sat_thresh", -1),
        ("sat_thresh", 256),
        ("thresh_lo", -1),
        ("thresh_hi", 256),
        ("ref_thresh", 256),
        ("min_area", -1),
        ("max_area", -1),
        ("max_aspect", -1),
        ("morph_kernel", -1),
        ("morph_iter", -1),
        ("bg_history", -1),
        ("bg_var", -1),
        ("split_area", -1),
        ("unit_area", -1),
        ("merge_dist", -1),
        ("track_ttl", -1),
        ("min_hits", 0),
        ("min_speed", -1),
        ("max_dist", 0),
        ("warmup", -1),
        ("line", -0.1),
        ("line", 1.1),
        ("line_band", -0.1),
        ("line_band", 0.6),
        ("min_area_frac", -0.1),
        ("min_area_frac", 1.1),
        ("max_area_frac", -0.1),
        ("max_area_frac", 1.1),
        ("ref_alpha", -0.1),
        ("ref_alpha", 1.1),
    ],
)
def test_validate_params_rejects_out_of_range_values(name, value):
    with pytest.raises(ValueError, match=name):
        validate_params(complete(**{name: value}))


def test_validate_params_rejects_reversed_thresholds():
    with pytest.raises(ValueError, match="thresh_lo"):
        validate_params(complete(thresh_lo=200, thresh_hi=100))


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        ({"min_area": 300, "max_area": 299}, "max_area"),
        ({"min_area_frac": 0.3, "max_area_frac": 0.2}, "max_area_frac"),
    ],
)
def test_validate_params_rejects_active_maximum_below_minimum(changes, name):
    with pytest.raises(ValueError, match=name):
        validate_params(complete(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"min_area": 300, "max_area": 0},
        {"min_area_frac": 0.3, "max_area_frac": 0},
    ],
)
def test_validate_params_allows_inactive_maximum(changes):
    params = complete(**changes)

    assert validate_params(params) is params


def test_validate_params_rejects_unknown_and_missing_complete_keys():
    unknown = complete(unknown=1)
    missing = complete()
    del missing["scale"]

    with pytest.raises(ValueError, match="unknown"):
        validate_params(unknown)
    with pytest.raises(ValueError, match="missing.*scale"):
        validate_params(missing)


def test_validate_params_checks_roi_when_dimensions_are_supplied():
    with pytest.raises(ValueError, match="roi"):
        validate_params(complete(roi="120,2,140,20"), width=100, height=80)


def test_parse_roi_accepts_string_and_integer_sequence():
    assert parse_roi("1,2,30,40", 100, 80) == (1, 2, 30, 40)
    assert parse_roi([1, 2, 30, 40], 100, 80) == (1, 2, 30, 40)
    assert parse_roi(None, 100, 80) is None


def test_parse_roi_clips_to_frame():
    assert parse_roi("-5,2,120,90", 100, 80) == (0, 2, 100, 80)


@pytest.mark.parametrize(
    "roi",
    [
        "1,2,3",
        "1,2,3,4,5",
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ],
)
def test_parse_roi_rejects_wrong_arity(roi):
    with pytest.raises(ValueError, match="roi"):
        parse_roi(roi, 100, 80)


@pytest.mark.parametrize(
    "roi",
    [
        "1,2,nope,4",
        [1, 2, 3.5, 4],
        [1, 2, "3", 4],
        [1, 2, True, 4],
        123,
    ],
)
def test_parse_roi_rejects_non_integer_values(roi):
    with pytest.raises(ValueError, match="roi"):
        parse_roi(roi, 100, 80)


def test_parse_roi_rejects_empty_clipped_region():
    with pytest.raises(ValueError, match="roi"):
        parse_roi("120,2,140,20", 100, 80)
