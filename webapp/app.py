#!/usr/bin/env python3
"""Flask backend for the conveyor-counter web viewer.

Wraps the ``annotate`` module: pick or upload a video, run detection/tracking
with any parameters, and get structured per-frame annotations, foreground
masks, and ground-truth metrics.  A per-(video,scale) frame cache and a
per-detection-signature cache let tracker-only parameter changes re-count
almost instantly (needed by the interactive ROI/line drawing).

Run:  python3 webapp/app.py   ->  http://127.0.0.1:5000
"""
import glob
import json
import os
import re
import subprocess
import sys
import uuid

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import annotate
from count_cv import DEFAULT_PARAMS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.expanduser("~/tmp")
RUNS_DIR = os.path.join(TMP, "inv_web_runs")
UPLOAD_DIR = os.path.join(TMP, "inv_web_uploads")
CACHE_ROOT = os.path.join(TMP, "inv_web_cache")
SAMPLE_DIRS = [os.path.join(REPO, "validation_scenarios"), UPLOAD_DIR, TMP]
ALLOWED_ROOTS = [os.path.realpath(p) for p in (REPO, TMP)]
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
MAX_FRAMES_CAP = 1500

app = Flask(__name__, static_folder="static", static_url_path="")
for d in (RUNS_DIR, UPLOAD_DIR, CACHE_ROOT):
    os.makedirs(d, exist_ok=True)


# ---- helpers ---------------------------------------------------------------

def _under_allowed(path):
    real = os.path.realpath(path)
    return any(real == root or real.startswith(root + os.sep) for root in ALLOWED_ROOTS)


def coerce_params(raw):
    """Cast web JSON values to the types expected by DEFAULT_PARAMS."""
    out = {}
    for key, val in (raw or {}).items():
        if key == "roi":
            if val:
                out["roi"] = [int(v) for v in str(val).split(",")][:4]
            continue
        if key not in DEFAULT_PARAMS or val is None or val == "":
            continue
        default = DEFAULT_PARAMS[key]
        if isinstance(default, bool):
            out[key] = val in (True, "true", "1", 1, "on")
        elif isinstance(default, int) and not isinstance(default, bool):
            out[key] = int(float(val))
        elif isinstance(default, float):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def _max_frames(body):
    """Clamp max_frames: absent/invalid -> safety cap; explicit <=0 -> 0 (all)."""
    raw = body.get("max_frames")
    if raw is None or raw == "":
        return MAX_FRAMES_CAP
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return MAX_FRAMES_CAP
    return 0 if v <= 0 else min(v, MAX_FRAMES_CAP)


def _valid_video(video):
    return video and os.path.isfile(video) and _under_allowed(video)


def _meta_gt(path):
    meta = annotate.find_meta(path)
    if not meta:
        return None
    try:
        with open(meta, encoding="utf-8") as fh:
            return json.load(fh).get("crossed_center_line")
    except (OSError, ValueError):
        return None


def _summary(result):
    return {k: result[k] for k in ("run_id", "meta", "resolved", "line", "count",
                                   "timing", "metrics", "frames_key", "det_key")
            if k in result} | {"frames_total": len(result["frames"])}


# ---- pages / media ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/videos")
def list_videos():
    seen, items = set(), []
    for d in SAMPLE_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "*"))):
            if path.lower().endswith(VIDEO_EXTS) and path not in seen and os.path.isfile(path):
                seen.add(path)
                items.append({"path": path, "name": os.path.basename(path),
                              "dir": os.path.basename(d.rstrip("/")), "gt": _meta_gt(path)})
    return jsonify({"videos": items, "defaults": DEFAULT_PARAMS})


@app.route("/api/frames/<key>/<name>")
def serve_frame(key, name):
    return send_from_directory(os.path.join(CACHE_ROOT, "frames", secure_filename(key)),
                               secure_filename(name), max_age=3600)


@app.route("/api/masks/<key>/<name>")
def serve_mask(key, name):
    return send_from_directory(os.path.join(CACHE_ROOT, "masks", secure_filename(key)),
                               secure_filename(name), max_age=3600)


@app.route("/api/run/<run_id>/annotations.json")
def run_annotations(run_id):
    path = os.path.join(RUNS_DIR, secure_filename(run_id), "annotations.json")
    if not os.path.isfile(path):
        return jsonify({"error": "run 不存在"}), 404
    return send_file(path, mimetype="application/json")


# ---- actions ---------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "缺少上传文件"}), 400
    name = secure_filename(file.filename)
    if not name.lower().endswith(VIDEO_EXTS):
        return jsonify({"error": "不支持的视频格式"}), 400
    dest = os.path.join(UPLOAD_DIR, name)
    file.save(dest)
    return jsonify({"path": dest, "name": name, "gt": _meta_gt(dest)})


@app.route("/api/run", methods=["POST"])
def run():
    body = request.get_json(force=True) or {}
    video = body.get("video")
    if not _valid_video(video):
        return jsonify({"error": "视频路径无效或不可访问"}), 400
    params = coerce_params(body.get("params"))
    max_frames = _max_frames(body)
    try:
        result = annotate.run_cached(video, params, float(body.get("fps") or 30.0),
                                     max_frames, CACHE_ROOT, start=int(body.get("start") or 0))
    except Exception as exc:   # noqa: BLE001 - surface engine errors to the UI
        return jsonify({"error": f"运行失败: {exc}"}), 500

    truth = annotate.load_truth(video, result["meta"]["width"], result["meta"]["height"],
                                result["meta"]["scale"], labels_dir=body.get("labels_dir"))
    result["metrics"] = annotate.frame_metrics(result, truth) if truth else None
    result["truth"] = truth
    run_id = uuid.uuid4().hex[:12]
    result["run_id"] = run_id
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "annotations.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    return jsonify(_summary(result))


@app.route("/api/batch", methods=["POST"])
def batch():
    body = request.get_json(force=True) or {}
    videos = [v for v in (body.get("videos") or []) if _valid_video(v)]
    if not videos:
        return jsonify({"error": "没有有效视频"}), 400
    params = coerce_params(body.get("params"))
    max_frames = _max_frames(body)
    rows = annotate.batch_run(videos, params, 30.0, max_frames, CACHE_ROOT, start=int(body.get("start") or 0))
    return jsonify({"rows": rows})


@app.route("/api/grid", methods=["POST"])
def grid():
    body = request.get_json(force=True) or {}
    video = body.get("video")
    if not _valid_video(video):
        return jsonify({"error": "视频路径无效"}), 400
    base = coerce_params(body.get("base_params"))
    raw_grid = body.get("grid") or {}
    grid_spec = {}
    for k, vals in raw_grid.items():
        if k not in DEFAULT_PARAMS or not isinstance(vals, list):
            continue
        grid_spec[k] = [coerce_params({k: v})[k] for v in vals if coerce_params({k: v})]
    if not grid_spec:
        return jsonify({"error": "网格为空或参数名无效"}), 400
    max_frames = _max_frames(body)
    try:
        out = annotate.grid_search(video, base, grid_spec, 30.0, max_frames, CACHE_ROOT, start=int(body.get("start") or 0))
    except Exception as exc:   # noqa: BLE001
        return jsonify({"error": f"网格搜索失败: {exc}"}), 500
    return jsonify(out)


@app.route("/api/autoparams", methods=["POST"])
def autoparams():
    body = request.get_json(force=True) or {}
    video = body.get("video")
    if not _valid_video(video):
        return jsonify({"error": "视频路径无效"}), 400
    try:
        return jsonify(annotate.auto_estimate(video))
    except Exception as exc:   # noqa: BLE001
        return jsonify({"error": f"自动估参失败: {exc}"}), 500


@app.route("/api/pytest", methods=["POST"])
def run_pytest():
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
                              capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return jsonify({"passed": 0, "failed": 0, "ok": False,
                        "output": "pytest 超时(>180s)"})
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    passed = failed = 0
    for line in tail:
        mp = re.search(r"(\d+) passed", line)
        mf = re.search(r"(\d+) failed", line)
        if mp:
            passed = int(mp.group(1))
        if mf:
            failed = int(mf.group(1))
    return jsonify({"passed": passed, "failed": failed, "ok": proc.returncode == 0,
                    "output": "\n".join(tail)})


@app.route("/api/validate", methods=["POST"])
def run_validate():
    body = request.get_json(force=True) or {}
    profile = body.get("profile", "smoke")
    if profile not in ("smoke", "quick", "full"):
        profile = "smoke"
    results = os.path.join(REPO, "validation_scenarios", "results.json")
    if body.get("run"):
        try:
            proc = subprocess.run([sys.executable, "validate_scenarios.py", "--profile",
                                   profile, "--mode", "auto", "--reuse"], cwd=REPO,
                                  capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return jsonify({"error": "验证运行超时(>600s)"}), 504
        if proc.returncode != 0:
            return jsonify({"error": "验证运行失败",
                            "output": (proc.stdout + proc.stderr)[-800:]}), 500
    if not os.path.isfile(results):
        return jsonify({"error": "尚无验证报告，请先运行"}), 404
    with open(results, encoding="utf-8") as fh:
        return jsonify(json.load(fh))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
