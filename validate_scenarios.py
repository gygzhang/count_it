#!/usr/bin/env python3
"""Generate a small scenario matrix and measure counting accuracy.

The harness intentionally validates *one object class* while changing the
quantities which previously required manual tuning: object size, speed and
FPS.  Every generated video has an accompanying ``*_meta.json`` ground truth.

Quick target-resolution run:

    python validate_scenarios.py --profile quick --mode auto

Compare the old fixed configuration with adaptive tracking:

    python validate_scenarios.py --profile quick --mode both --reuse

Results are written as JSON, CSV and a Markdown table so a regression is easy
to inspect or archive.  ``--reuse`` avoids regenerating videos which already
exist and is useful while iterating on the detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from count_cv import count_source


@dataclass(frozen=True)
class Scenario:
    name: str
    fps: int
    speed: float
    min_size: float
    max_size: float
    count: int = 12
    scale_jitter: float = 0.25
    rotation_jitter: float = 3.0
    wobble: float = 0.15

    @property
    def step_px(self) -> float:
        return self.speed / self.fps


def scenario_matrix(profile: str) -> list[Scenario]:
    """Return a compact one-factor-at-a-time matrix plus one hard combination."""
    if profile == "smoke":
        # Kept tiny for CI and script checks, rather than performance claims.
        return [
            Scenario("baseline", 60, 1200, 14, 22, count=5),
            Scenario("fast_small", 60, 2400, 9, 14, count=5),
        ]
    if profile == "full":
        return [
            Scenario("small", 300, 12000, 12, 20),
            Scenario("medium", 300, 12000, 25, 35),
            Scenario("large", 300, 12000, 40, 55),
            Scenario("slow", 300, 3000, 25, 35),
            Scenario("medium_speed", 300, 6000, 25, 35),
            Scenario("fast", 300, 12000, 25, 35),
            Scenario("fps_60", 60, 2400, 25, 35),
            Scenario("fps_120", 120, 4800, 25, 35),
            Scenario("fps_300", 300, 12000, 25, 35),
            Scenario("hard_combo", 120, 7200, 12, 20, count=20,
                     scale_jitter=0.35, rotation_jitter=5.0, wobble=0.20),
        ]
    # Quick still runs at the requested 800x500 resolution.  These scenarios
    # isolate size, speed and FPS without producing several minutes of video.
    return [
        Scenario("baseline", 120, 4800, 25, 35),
        Scenario("small", 120, 4800, 12, 20),
        Scenario("large", 120, 4800, 40, 55),
        Scenario("slow", 120, 2400, 25, 35),
        Scenario("fast", 120, 7200, 25, 35),
        Scenario("low_fps", 60, 2400, 25, 35),
        Scenario("high_fps", 300, 12000, 25, 35),
        Scenario("hard_combo", 120, 7200, 12, 20, count=20,
                 scale_jitter=0.35, rotation_jitter=5.0, wobble=0.20),
    ]


def generator_command(s: Scenario, video: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).with_name("gen_shapes_video.py")),
        "-o", str(video),
        "--duration", str(args.duration),
        "--fps", str(s.fps),
        "--width", str(args.width),
        "--height", str(args.height),
        "--count", str(s.count),
        "--speed", str(s.speed),
        "--min-size", str(s.min_size),
        "--max-size", str(s.max_size),
        "--scale-jitter", str(s.scale_jitter),
        "--rotation-jitter", str(s.rotation_jitter),
        "--wobble", str(s.wobble),
        "--seed", str(args.seed),
        "--gray",
    ]
    if args.assets:
        cmd += ["--twemoji-assets", str(args.assets),
                "--twemoji-object", args.object]
    return cmd


def fixed_params(direction: str) -> dict:
    """The formerly hand-tuned Hammer settings, used only as a comparison."""
    return {
        "method": "thresh", "thresh_lo": 90, "thresh_hi": 150, "min_area": 20,
        "morph_kernel": 5, "morph_iter": 1,
        "max_dist": 50, "track_ttl": 5, "min_hits": 2,
        "axis": "x", "flow": "pos" if direction == "l2r" else "neg",
        "scale": 1.0,
    }


def count_params(mode: str) -> dict:
    params = fixed_params("l2r")
    if mode == "auto":
        # count_cv owns adaptation; grayscale foreground uses intensity
        # thresholding rather than color saturation.
        params["auto_adapt"] = True
    return params


def read_ground_truth(video: Path) -> tuple[int, dict]:
    meta_path = video.with_name(video.stem + "_meta.json")
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    return int(meta["crossed_center_line"]), meta


def result_row(s: Scenario, mode: str, detected: int, gt: int,
               elapsed: float, video: Path) -> dict:
    error = detected - gt
    abs_pct = (100.0 * abs(error) / gt) if gt else (0.0 if not error else float("inf"))
    return {
        "scenario": s.name,
        "mode": mode,
        "fps": s.fps,
        "speed_px_s": s.speed,
        "step_px_frame": round(s.step_px, 3),
        "diameter_px": f"{2*s.min_size:g}-{2*s.max_size:g}",
        "objects_on_screen": s.count,
        "ground_truth": gt,
        "detected": detected,
        "error": error,
        "absolute_error_pct": round(abs_pct, 3),
        "count_accuracy_pct": round(max(0.0, 100.0 - abs_pct), 3),
        "detect_seconds": round(elapsed, 3),
        "video": str(video),
    }


def write_reports(rows: list[dict], prefix: Path, config: dict) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump({"config": config, "results": rows}, fh,
                  ensure_ascii=False, indent=2)
    if rows:
        with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    columns = [
        "scenario", "mode", "fps", "speed_px_s", "step_px_frame",
        "diameter_px", "ground_truth", "detected", "error",
        "absolute_error_pct", "count_accuracy_pct",
    ]
    with prefix.with_suffix(".md").open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row[c]) for c in columns) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="多尺寸/速度/FPS 自动适配准确率验证")
    ap.add_argument("--profile", choices=["smoke", "quick", "full"], default="quick")
    ap.add_argument("--mode", choices=["auto", "fixed", "both"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=Path("validation_scenarios"))
    ap.add_argument("--report", type=Path, default=None,
                    help="报告前缀；默认 OUT_DIR/results")
    ap.add_argument("--assets", type=Path, default=Path("twemoji_assets"))
    ap.add_argument("--object", default="hammer")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.duration is None:
        args.duration = 0.6 if args.profile == "smoke" else (1.25 if args.profile == "quick" else 2.5)
    if args.width is None:
        args.width = 320 if args.profile == "smoke" else 800
    if args.height is None:
        args.height = 200 if args.profile == "smoke" else 500
    if args.assets and not args.assets.is_dir():
        raise SystemExit(f"Twemoji 资源目录不存在: {args.assets}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.report or args.out_dir / "results"
    modes = ["fixed", "auto"] if args.mode == "both" else [args.mode]
    rows: list[dict] = []
    scenarios = scenario_matrix(args.profile)
    config = {
        "profile": args.profile, "resolution": [args.width, args.height],
        "duration_s": args.duration, "object": args.object,
        "modes": modes, "scenarios": [asdict(s) for s in scenarios],
    }

    for i, scenario in enumerate(scenarios, 1):
        video = args.out_dir / f"{scenario.name}.mp4"
        meta = video.with_name(video.stem + "_meta.json")
        if not args.reuse or not video.exists() or not meta.exists():
            print(f"[{i}/{len(scenarios)}] 生成 {scenario.name} ...", flush=True)
            run = subprocess.run(generator_command(scenario, video, args),
                                 text=True, capture_output=not args.verbose)
            if run.returncode:
                if run.stdout:
                    print(run.stdout, file=sys.stderr)
                if run.stderr:
                    print(run.stderr, file=sys.stderr)
                raise SystemExit(f"生成失败: {scenario.name}")
        gt, _ = read_ground_truth(video)
        for mode in modes:
            print(f"[{i}/{len(scenarios)}] 检测 {scenario.name} ({mode}) ...",
                  flush=True)
            started = time.perf_counter()
            detected = count_source(str(video), count_params(mode), verbose=args.verbose)
            row = result_row(scenario, mode, detected, gt,
                             time.perf_counter() - started, video)
            rows.append(row)
            print(f"  GT={gt} DET={detected} error={row['error']:+d} "
                  f"accuracy={row['count_accuracy_pct']:.2f}%")
            # Preserve partial results if a long matrix is interrupted.
            write_reports(rows, report, config)

    print("\n| 场景 | 模式 | GT | 检测 | 误差 | 准确率 |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['scenario']} | {row['mode']} | {row['ground_truth']} | "
              f"{row['detected']} | {row['error']:+d} | "
              f"{row['count_accuracy_pct']:.2f}% |")
    print(f"\n报告: {report.with_suffix('.md')} / .csv / .json")


if __name__ == "__main__":
    main()
