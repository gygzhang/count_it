from argparse import Namespace
from pathlib import Path

from validate_scenarios import (
    Scenario,
    generator_command,
    result_row,
    scenario_matrix,
)


def test_quick_matrix_varies_size_speed_and_fps():
    scenarios = scenario_matrix("quick")
    assert len({(s.min_size, s.max_size) for s in scenarios}) > 1
    assert len({s.speed for s in scenarios}) > 1
    assert len({s.fps for s in scenarios}) > 1


def test_generator_command_carries_scenario_parameters():
    scenario = Scenario("case", 120, 4800, 12, 20)
    args = Namespace(duration=1.0, width=800, height=500, seed=7,
                     assets=Path("twemoji_assets"), object="hammer")
    command = generator_command(scenario, Path("case.mp4"), args)
    joined = " ".join(command)
    assert "--fps 120" in joined
    assert "--speed 4800" in joined
    assert "--min-size 12" in joined
    assert "--twemoji-object hammer" in joined


def test_result_reports_signed_error_and_absolute_accuracy():
    row = result_row(Scenario("case", 100, 4000, 10, 20),
                     "auto", detected=95, gt=100, elapsed=0.25,
                     video=Path("case.mp4"))
    assert row["step_px_frame"] == 40
    assert row["error"] == -5
    assert row["absolute_error_pct"] == 5
    assert row["count_accuracy_pct"] == 95
