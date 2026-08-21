#!/usr/bin/env python3
"""Scenario matrix for grayscale rigid-object counting.

Outputs all artifacts under ~/d3_tmp by default.  Generates videos with one
factor changed at a time (size, count, speed, FPS, direction, contrast, noise,
blur) and runs count_cv with --auto-adapt.  Reports CSV/JSON/Markdown.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from count_cv import count_source

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Case:
    name: str
    fps: int = 120
    speed: float = 2400
    count: int = 10
    min_size: float = 20
    max_size: float = 24
    direction: str = "l2r"
    contrast: float = 80
    polarity: str = "bright"
    noise: float = 0
    blur: int = 0
    seed: int = 20260819

    @property
    def axis(self): return "x" if self.direction in ("l2r", "r2l") else "y"
    @property
    def flow(self): return "pos" if self.direction in ("l2r", "t2b") else "neg"
    @property
    def step(self): return self.speed / self.fps


def cases(profile: str):
    base = Case("baseline")
    if profile == "smoke":
        return [base, Case("small", min_size=12, max_size=14),
                Case("fast", speed=4800), Case("vertical", direction="t2b"),
                Case("noise_blur", noise=5, blur=3)]
    # Compact matrix: one-factor cases plus interactions most relevant to the
    # tracker. Keep generated videos short, while retaining 1s start/end empty.
    return [
        base,
        Case("size_small", min_size=12, max_size=14),
        Case("size_large", min_size=32, max_size=36),
        Case("count_1", count=1), Case("count_20", count=20),
        Case("speed_slow", speed=1200), Case("speed_medium", speed=3600),
        Case("speed_fast", speed=6000), Case("speed_extreme", speed=7200),
        Case("fps_60", fps=60, speed=1200), Case("fps_300", fps=300, speed=6000),
        Case("r2l", direction="r2l"), Case("t2b", direction="t2b"),
        Case("b2t", direction="b2t"),
        Case("contrast_low", contrast=30), Case("contrast_high", contrast=110),
        Case("dark", polarity="dark"), Case("mixed", polarity="mixed"),
        Case("noise_3", noise=3), Case("noise_10", noise=10),
        Case("blur_3", blur=3), Case("blur_7", blur=7),
        Case("near_fast", count=20, speed=6000, min_size=12, max_size=14),
    ]


def generate(case: Case, video: Path, args):
    cmd = [sys.executable, str(ROOT / "gen_shapes_video.py"), "-o", str(video),
           "--duration", str(args.duration), "--fps", str(case.fps),
           "--width", str(args.width), "--height", str(args.height),
           "--count", str(case.count), "--speed", str(case.speed),
           "--direction", case.direction, "--min-size", str(case.min_size),
           "--max-size", str(case.max_size), "--gray", "--bg-gray", "128",
           "--gray-contrast", str(case.contrast), "--gray-polarity", case.polarity,
           "--noise", str(case.noise), "--motion-blur", str(case.blur),
           "--empty-start", "1", "--empty-end", "1", "--seed", str(case.seed)]
    if args.assets and args.assets.is_dir():
        cmd += ["--twemoji-assets", str(args.assets), "--twemoji-object", args.object]
    run = subprocess.run(cmd, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(f"generation failed {case.name}: {run.stderr[-1000:]}")


def run_case(case: Case, out: Path, args):
    video = out / f"{case.name}.mp4"
    meta = video.with_name(video.stem + "_meta.json")
    if not args.reuse or not video.exists() or not meta.exists():
        generate(case, video, args)
    with meta.open(encoding="utf-8") as f: m = json.load(f)
    gt = int(m["crossed_center_line"])
    params = {"method":"thresh", "thresh_lo":90, "thresh_hi":150,
              "auto_adapt":True, "axis":case.axis, "flow":case.flow,
              "calibration_frames": min(args.calibration_frames, max(1, int(case.fps))),
              "scale":1.0}
    t = time.perf_counter(); detected = count_source(str(video), params, verbose=False)
    elapsed = time.perf_counter() - t
    err = detected - gt
    return {"scenario":case.name, "fps":case.fps, "speed_px_s":case.speed,
            "step_px_frame":round(case.step,3), "count":case.count,
            "direction":case.direction, "min_size":case.min_size, "max_size":case.max_size,
            "contrast":case.contrast, "polarity":case.polarity, "noise":case.noise,
            "blur":case.blur, "ground_truth":gt, "detected":detected, "error":err,
            "accuracy_pct":round(100*max(0, 1-abs(err)/gt),3) if gt else 100.0,
            "detect_seconds":round(elapsed,3), "video":str(video)}


def reports(rows, prefix, config):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps({"config":config,"results":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    if rows:
        # Error rows may contain diagnostic fields absent from successful rows.
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with prefix.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    cols=["scenario","fps","speed_px_s","step_px_frame","count","direction","ground_truth","detected","error","accuracy_pct"]
    with prefix.with_suffix(".md").open("w",encoding="utf-8") as f:
        f.write("|"+"|".join(cols)+"|\n|"+"|".join(["---"]*len(cols))+"|\n")
        for r in rows: f.write("|"+"|".join(str(r[c]) for c in cols)+"|\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--profile",choices=["smoke","full"],default="smoke")
    ap.add_argument("--out-dir",type=Path,default=Path.home()/"d3_tmp"/"scenario_matrix")
    ap.add_argument("--duration",type=float,default=3.0); ap.add_argument("--width",type=int,default=800); ap.add_argument("--height",type=int,default=500)
    ap.add_argument("--assets",type=Path,default=ROOT/"twemoji_assets"); ap.add_argument("--object",default="hammer")
    ap.add_argument("--calibration-frames",type=int,default=48); ap.add_argument("--seed",type=int,default=20260819); ap.add_argument("--reuse",action="store_true")
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    cs=[c for c in cases(args.profile)]; rows=[]
    for i,c in enumerate(cs,1):
        print(f"[{i}/{len(cs)}] {c.name}",flush=True)
        try: r=run_case(c,args.out_dir,args); rows.append(r); print(f"  GT={r['ground_truth']} DET={r['detected']} err={r['error']:+d} acc={r['accuracy_pct']:.2f}%",flush=True)
        except Exception as e: print(f"  ERROR: {e}",file=sys.stderr); rows.append({"scenario":c.name,"error":"exception","detail":str(e)})
        reports(rows,args.out_dir/"results",{"profile":args.profile,"cases":[asdict(x) for x in cs]})
    print(f"reports: {args.out_dir/'results.md'}")

if __name__=='__main__': main()
