import json

import pytest

from count_cv import resolve_cli_params
from params import DEFAULT_PARAMS


def test_params_file_fills_parameters_when_cli_is_absent(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"max_dist": 77, "min_hits": 3}))

    source, params, output = resolve_cli_params(
        ["sample.mp4", "--params", str(path)]
    )

    assert source == "sample.mp4"
    assert params["max_dist"] == 77
    assert params["min_hits"] == 3
    assert params["method"] == DEFAULT_PARAMS["method"]
    assert output.fps == 30.0


def test_explicit_cli_parameter_overrides_params_file(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"max_dist": 77}))

    _, params, _ = resolve_cli_params(
        ["sample.mp4", "--params", str(path), "--max-dist", "99"]
    )

    assert params["max_dist"] == 99.0


def test_invalid_json_parameter_is_an_argparse_error(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"max_dist": 0}))

    with pytest.raises(SystemExit) as exc:
        resolve_cli_params(["sample.mp4", "--params", str(path)])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "path, contents",
    [
        ("missing.json", None),
        ("malformed.json", "{not json"),
    ],
)
def test_params_file_read_errors_are_argparse_errors(tmp_path, path, contents):
    path = tmp_path / path
    if contents is not None:
        path.write_text(contents)

    with pytest.raises(SystemExit) as exc:
        resolve_cli_params(["sample.mp4", "--params", str(path)])

    assert exc.value.code == 2
