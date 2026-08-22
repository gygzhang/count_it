"""Structured per-frame annotation export for the web viewer.

Reuses the ``count_cv`` engine (decode -> resolve method -> optional
auto-adapt -> Detector/Tracker) but *collects* per-frame detections, track
states, crossing events and running counts as plain data instead of drawing
them into pixels.  It also dumps the processed frames as JPEGs so the browser
renders overlays on a <canvas> without depending on the annotated-video codec.

Ground-truth (generator per-frame boxes + sidecar total) is loaded when
available and matched against detections to produce simple accuracy metrics.
"""
import glob
import hashlib
import itertools
import json
import os
import re
import shutil
import threading
import time

import cv2
import numpy as np

from count_cv import (DEFAULT_PARAMS, DET_KEYS, Detector, FrameSource, Tracker,
                      auto_adapt_params, is_warming, resolve_method, scaled)


def _det_box(d):
    """Detection tuple (cx,cy,x,y,w,h,mult) -> integer [x,y,w,h]."""
    return [int(round(d[2])), int(round(d[3])), int(round(d[4])), int(round(d[5]))]


def annotate_source(source, params=None, fps=30.0, max_frames=0,
                    dump_frames=None, jpg_quality=80):
    """Run detection + tracking on a file/folder, returning structured data.

    Returns a dict with ``meta``, ``resolved`` (method/band/auto-adapt diag/
    effective params), ``line``, ``count``, ``timing`` and a ``frames`` list of
    ``{i, dets, tracks, events, count}``.  Coordinates are in the *processed*
    resolution (after ``scale``); dumped frames share that resolution.
    """
    P = {**DEFAULT_PARAMS, **(params or {})}
    scale = float(P.get("scale", 1.0) or 1.0)

    src = FrameSource(source, fps)
    src_fps = float(src.fps or fps)
    w, h = int(src.w * scale), int(src.h * scale)
    frames = [scaled(f, scale, w, h) for f in src.frames()]
    src.release()
    if not frames:
        raise RuntimeError("无法从输入解码任何帧")
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]

    # Files are seekable, so the live-camera buffering in count_source is not
    # needed here; resolve the method (auto may pick thresh and write its band
    # into P) then optionally calibrate pixel parameters.
    method, ref = resolve_method(P, frames, w, h)
    diag = {}
    if P.get("auto_adapt"):
        P, diag = auto_adapt_params(P, frames, method, w, h, ref, fps=src_fps)

    det = Detector(P, method, w, h, ref)
    trk = Tracker(P, w, h)

    if dump_frames:
        os.makedirs(dump_frames, exist_ok=True)

    proc_ms = []
    out_frames = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        dets = det.detect(frame)
        trk.update(dets, is_warming(method, i, P["warmup"]))
        proc_ms.append((time.perf_counter() - t0) * 1000.0)

        out_frames.append({
            "i": i,
            "dets": [_det_box(d) for d in dets],
            "tracks": [{"id": int(t.id),
                        "cx": round(float(t.cx), 1), "cy": round(float(t.cy), 1),
                        "counted": bool(t.counted),
                        "mult": int(getattr(t, "multiplicity", 1)),
                        "hits": int(t.hits), "missing": int(t.missing)}
                       for t in trk.tracks],
            "events": [int(t.id) for t in trk.tracks if t.cross_frame == i],
            "count": int(trk.count),
        })
        if dump_frames:
            cv2.imwrite(os.path.join(dump_frames, f"frame_{i:06d}.jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])

    a = np.asarray(proc_ms, dtype=float)
    mean = float(a.mean())
    resolved = {
        "method": method,
        "auto_adapt": bool(P.get("auto_adapt")),
        "diag": diag,
        "params": {k: P[k] for k in DEFAULT_PARAMS},
    }
    if method == "thresh":
        resolved["thresh_lo"] = int(P["thresh_lo"])
        resolved["thresh_hi"] = int(P["thresh_hi"])
    return {
        "meta": {"source": str(source), "width": w, "height": h,
                 "src_fps": round(src_fps, 3), "frames": len(frames),
                 "scale": scale},
        "resolved": resolved,
        "line": {"axis": trk.axis, "pos": int(trk.line_pos),
                 "band": float(trk.band)},
        "count": int(trk.count),
        "timing": {"avg_ms": round(mean, 3),
                   "median_ms": round(float(np.median(a)), 3),
                   "p95_ms": round(float(np.percentile(a, 95)), 3),
                   "max_ms": round(float(a.max()), 3),
                   "throughput_fps": round(1000.0 / mean, 1) if mean > 0 else None},
        "frames": out_frames,
    }


def find_meta(source):
    """Locate the generator sidecar *_meta.json for a video or image folder."""
    if os.path.isdir(source):
        cands = sorted(glob.glob(os.path.join(source, "*_meta.json")))
        return cands[0] if cands else None
    cand = os.path.splitext(source)[0] + "_meta.json"
    return cand if os.path.exists(cand) else None


def load_truth(source, w, h, scale=1.0, labels_dir=None):
    """Load generator ground-truth: sidecar total + optional per-frame boxes.

    Returns ``{gt_total, per_frame}`` where ``per_frame[i]`` is a list of
    ``[x,y,w,h]`` boxes in the *processed* resolution, or ``None`` when no
    ground truth is available.  Per-frame boxes come from the richer
    ``frame_XXXXXX.json`` when present, else YOLO ``.txt``.
    """
    meta_path = find_meta(source)
    gt_total = None
    if meta_path:
        try:
            with open(meta_path, encoding="utf-8") as fh:
                gt_total = int(json.load(fh).get("crossed_center_line"))
        except (OSError, ValueError, TypeError):
            gt_total = None

    # Per-frame truth boxes are optional and only present when the clip was
    # generated with --labels.  Accept an explicit dir, an image-folder source,
    # or a "<video_stem>_labels" sibling directory.
    if labels_dir is None:
        if os.path.isdir(source):
            labels_dir = source
        else:
            guess = os.path.splitext(source)[0] + "_labels"
            labels_dir = guess if os.path.isdir(guess) else None
    per_frame = None
    if labels_dir and os.path.isdir(labels_dir):
        per_frame = _load_label_dir(labels_dir, w, h, scale)

    if gt_total is None and per_frame is None:
        return None
    return {"gt_total": gt_total, "per_frame": per_frame}


def _load_label_dir(labels_dir, w, h, scale):
    jsons = sorted(glob.glob(os.path.join(labels_dir, "frame_*.json")))
    txts = sorted(glob.glob(os.path.join(labels_dir, "frame_*.txt")))
    out = {}
    for path in jsons:
        idx = _frame_index(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        boxes = []
        for o in data.get("objects", []):
            x0, y0, x1, y1 = o["bbox_xyxy"]
            boxes.append([int(x0 * scale), int(y0 * scale),
                          int((x1 - x0) * scale), int((y1 - y0) * scale)])
        out[idx] = boxes
    if not out:  # fall back to YOLO txt (normalized) when no json labels
        for path in txts:
            idx = _frame_index(path)
            boxes = []
            for line in open(path, encoding="utf-8"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = (float(v) for v in parts)
                boxes.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int(bw * w), int(bh * h)])
            out[idx] = boxes
    return out


def _frame_index(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def frame_metrics(result, truth, iou_thresh=0.3):
    """Greedy IoU match of detections vs per-frame truth boxes.

    Returns aggregate ``{tp, fp, fn, precision, recall, count, gt_total,
    count_error}``; ``None`` when no per-frame truth is available.  Count-vs-GT
    (crossings) is reported regardless of per-frame boxes.
    """
    gt_total = truth.get("gt_total") if truth else None
    summary = {"count": result["count"], "gt_total": gt_total,
               "count_error": (result["count"] - gt_total)
               if gt_total is not None else None}
    per_frame = truth.get("per_frame") if truth else None
    if not per_frame:
        return summary

    tp = fp = fn = 0
    for fr in result["frames"]:
        gts = list(per_frame.get(fr["i"], []))
        dets = list(fr["dets"])
        used = [False] * len(gts)
        for d in dets:
            best, bj = iou_thresh, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = _iou(d, g)
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used[bj] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    summary.update({"tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 4), "recall": round(rec, 4)})
    return summary


# ---------------------------------------------------------------------------
# Cached, staged pipeline for the web app: dump frames once per (video, scale),
# cache the detection sequence + foreground masks per detection signature, and
# re-run only the (cheap) tracker when tracker-only parameters change.
# ---------------------------------------------------------------------------

# Tracker-side parameters that auto_adapt derives from calibration; kept as a
# baseline so a cached detection can be re-tracked while explicit user tracker
# parameters still win.
AUTO_TRK_KEYS = ("max_dist", "global_vx", "global_vy", "ordered_match",
                 "transverse_gate", "area_ratio_max", "shape_cost_weight",
                 "track_ttl", "min_hits", "cross_dedup_frames", "cross_dedup_dist")

_FRAMES = {}   # frame_key -> {"dir", "w", "h", "src_fps", "n"}
_DETS = {}     # det_key -> {"dets_seq", "mask_dir", "method", "band", "diag",
               #             "auto_trk", "det_ms", "w", "h"}
_MAX_FRAME_SETS = 12
_MAX_DET_SETS = 40
_LOCK = threading.RLock()   # serialize cache population/eviction (Flask threaded=True)


def _hash(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _decode(video, scale, fps, max_frames):
    src = FrameSource(video, fps)
    src_fps = float(src.fps or fps)
    w, h = int(src.w * scale), int(src.h * scale)
    frames = []
    for f in src.frames():
        frames.append(scaled(f, scale, w, h))
        if max_frames and len(frames) >= max_frames:
            break
    src.release()
    if not frames:
        raise RuntimeError("无法从输入解码任何帧")
    return frames, w, h, src_fps


def _det_sig(P):
    return _hash(json.dumps({k: (list(P[k]) if isinstance(P.get(k), tuple) else P.get(k))
                             for k in sorted(DET_KEYS)}, default=str))


def _drop_dets_for(fkey):
    """Evict detection entries built on a frame set, cleaning their masks."""
    for dk in [k for k, e in _DETS.items() if e.get("fkey") == fkey]:
        entry = _DETS.pop(dk)
        if entry.get("mask_dir") and os.path.isdir(entry["mask_dir"]):
            shutil.rmtree(entry["mask_dir"], ignore_errors=True)


def _lru_frames():
    # Evict oldest frame sets AND their dependent detections (keep them coherent).
    while len(_FRAMES) > _MAX_FRAME_SETS:
        key = next(iter(_FRAMES))
        entry = _FRAMES.pop(key)
        _drop_dets_for(key)
        if entry.get("dir") and os.path.isdir(entry["dir"]):
            shutil.rmtree(entry["dir"], ignore_errors=True)


def _lru_dets():
    while len(_DETS) > _MAX_DET_SETS:
        key = next(iter(_DETS))
        entry = _DETS.pop(key)
        if entry.get("mask_dir") and os.path.isdir(entry["mask_dir"]):
            shutil.rmtree(entry["mask_dir"], ignore_errors=True)


def _paste_mask(mask, w, h, roi):
    if roi is None or (mask.shape[1] == w and mask.shape[0] == h):
        return mask
    full = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = roi
    full[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
    return full


def ensure_frames(video, scale, fps, max_frames, cache_root):
    """Decode + dump processed frames once; returns (key, info, frames_or_None).

    Key is scoped to path+mtime+scale+max_frames+fps+cache_root so different
    fps (image folders), cache roots, or edited files never collide."""
    key = _hash(os.path.realpath(video), os.path.getmtime(video), scale,
                max_frames, round(float(fps), 3), os.path.realpath(cache_root))
    with _LOCK:
        if key in _FRAMES:
            return key, _FRAMES[key], None
        frames, w, h, src_fps = _decode(video, scale, fps, max_frames)
        fdir = os.path.join(cache_root, "frames", key)
        os.makedirs(fdir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(fdir, f"frame_{i:06d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        _FRAMES[key] = {"dir": fdir, "w": w, "h": h, "src_fps": src_fps, "n": len(frames)}
        _lru_frames()
        return key, _FRAMES[key], frames


def ensure_detection(fkey, finfo, frames, video, scale, fps, max_frames, P, cache_root):
    """Resolve method + optional auto-adapt + detect + dump masks (cached)."""
    w, h, src_fps = finfo["w"], finfo["h"], finfo["src_fps"]
    dkey = _hash(fkey, _det_sig(P))          # fkey already scopes fps/cache_root
    with _LOCK:
        if dkey in _DETS:
            return dkey, _DETS[dkey]
        if frames is None:
            frames, w, h, src_fps = _decode(video, scale, fps, max_frames)
        method, ref = resolve_method(P, frames, w, h)   # may write thresh band into P
        band = [int(P["thresh_lo"]), int(P["thresh_hi"])] if method == "thresh" else None
        diag, Pdet = {}, P
        if P.get("auto_adapt"):
            Pdet, diag = auto_adapt_params(P, frames, method, w, h, ref, fps=src_fps)
        auto_trk = {k: Pdet[k] for k in AUTO_TRK_KEYS if k in Pdet} if P.get("auto_adapt") else {}
        det = Detector(Pdet, method, w, h, ref)
        mask_dir = os.path.join(cache_root, "masks", dkey)
        os.makedirs(mask_dir, exist_ok=True)
        dets_seq, det_ms = [], []
        for i, f in enumerate(frames):
            t0 = time.perf_counter()
            dets = det.detect(f)
            det_ms.append((time.perf_counter() - t0) * 1000.0)
            dets_seq.append(dets)
            cv2.imwrite(os.path.join(mask_dir, f"frame_{i:06d}.jpg"),
                        _paste_mask(det.last_mask, w, h, det.roi),
                        [cv2.IMWRITE_JPEG_QUALITY, 70])
        _DETS[dkey] = {"dets_seq": dets_seq, "mask_dir": mask_dir, "fkey": fkey,
                       "method": method, "band": band, "diag": diag,
                       "auto_trk": auto_trk, "det_ms": det_ms, "w": w, "h": h}
        _lru_dets()
        return dkey, _DETS[dkey]


def _track_frames(dets_seq, P, w, h, method):
    trk = Tracker(P, w, h)
    out, track_ms = [], []
    for i, dets in enumerate(dets_seq):
        t0 = time.perf_counter()
        trk.update(dets, is_warming(method, i, P["warmup"]))
        track_ms.append((time.perf_counter() - t0) * 1000.0)
        out.append({
            "i": i,
            "dets": [_det_box(d) for d in dets],
            "tracks": [{"id": int(t.id), "cx": round(float(t.cx), 1),
                        "cy": round(float(t.cy), 1), "counted": bool(t.counted),
                        "mult": int(getattr(t, "multiplicity", 1)),
                        "vx": round(float(t.vx), 1), "vy": round(float(t.vy), 1),
                        "hits": int(t.hits), "missing": int(t.missing)}
                       for t in trk.tracks],
            "events": [int(t.id) for t in trk.tracks if t.cross_frame == i],
            "count": int(trk.count),
        })
    return out, int(trk.count), track_ms, trk


def run_cached(video, params=None, fps=30.0, max_frames=0, cache_root="."):
    """Cached staged run for the web app. Reuses dumped frames + cached
    detections so tracker-only parameter changes re-count almost instantly."""
    P = {**DEFAULT_PARAMS, **(params or {})}
    scale = float(P.get("scale", 1.0) or 1.0)
    fkey, finfo, frames = ensure_frames(video, scale, fps, max_frames, cache_root)
    dkey, dc = ensure_detection(fkey, finfo, frames, video, scale, fps,
                                max_frames, dict(P), cache_root)
    w, h = dc["w"], dc["h"]
    Ptrk = {**DEFAULT_PARAMS, **dc["auto_trk"], **(params or {})}
    frames_data, count, track_ms, trk = _track_frames(dc["dets_seq"], Ptrk, w, h, dc["method"])
    per = np.asarray(dc["det_ms"]) + np.asarray(track_ms)
    mean = float(per.mean()) if len(per) else 0.0
    resolved = {"method": dc["method"], "auto_adapt": bool(P.get("auto_adapt")),
                "diag": dc["diag"], "params": {k: Ptrk.get(k, P.get(k)) for k in DEFAULT_PARAMS}}
    if dc["band"]:
        resolved["thresh_lo"], resolved["thresh_hi"] = dc["band"]
    return {
        "meta": {"source": str(video), "width": w, "height": h,
                 "src_fps": round(finfo["src_fps"], 3), "frames": finfo["n"], "scale": scale},
        "resolved": resolved,
        "line": {"axis": trk.axis, "pos": int(trk.line_pos), "band": float(trk.band)},
        "count": count,
        "timing": {"avg_ms": round(mean, 3),
                   "median_ms": round(float(np.median(per)), 3) if len(per) else 0.0,
                   "p95_ms": round(float(np.percentile(per, 95)), 3) if len(per) else 0.0,
                   "max_ms": round(float(per.max()), 3) if len(per) else 0.0,
                   "throughput_fps": round(1000.0 / mean, 1) if mean > 0 else None},
        "frames": frames_data, "frames_key": fkey, "det_key": dkey, "has_mask": True,
    }


def batch_run(videos, params=None, fps=30.0, max_frames=0, cache_root="."):
    """Run one parameter set across several videos; returns a comparison list."""
    rows = []
    for v in videos:
        try:
            res = run_cached(v, params, fps, max_frames, cache_root)
            truth = load_truth(v, res["meta"]["width"], res["meta"]["height"], res["meta"]["scale"])
            m = frame_metrics(res, truth) if truth else None
            rows.append({"video": os.path.basename(v), "path": v, "ok": True,
                         "method": res["resolved"]["method"], "count": res["count"],
                         "gt": (m or {}).get("gt_total"), "error": (m or {}).get("count_error"),
                         "precision": (m or {}).get("precision"), "recall": (m or {}).get("recall"),
                         "avg_ms": res["timing"]["avg_ms"], "run_id": res["det_key"]})
        except Exception as exc:   # noqa: BLE001
            rows.append({"video": os.path.basename(v), "path": v, "ok": False, "error": str(exc)})
    return rows


def grid_search(video, base_params, grid, fps=30.0, max_frames=0, cache_root="."):
    """Small grid search on one video, ranked by |count-error| then SAE-style.

    ``grid`` maps parameter name -> list of candidate values.  Tracker-only
    grids reuse a single cached detection, so the sweep is fast."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rows = []
    for combo in combos:
        params = {**(base_params or {}), **dict(zip(keys, combo))}
        res = run_cached(video, params, fps, max_frames, cache_root)
        truth = load_truth(video, res["meta"]["width"], res["meta"]["height"], res["meta"]["scale"])
        m = frame_metrics(res, truth) if truth else None
        rows.append({"combo": dict(zip(keys, combo)), "count": res["count"],
                     "gt": (m or {}).get("gt_total"), "error": (m or {}).get("count_error"),
                     "abs_error": abs((m or {}).get("count_error")) if m and m.get("count_error") is not None else None,
                     "precision": (m or {}).get("precision"), "recall": (m or {}).get("recall"),
                     "avg_ms": res["timing"]["avg_ms"]})
    rows.sort(key=lambda r: (r["abs_error"] if r["abs_error"] is not None else 1e9))
    return {"keys": keys, "rows": rows}


def auto_estimate(video):
    """Run the no-truth initialiser (auto_params.estimate) for suggested params."""
    import auto_params
    params, diag = auto_params.estimate(video)
    return {"suggested": params, "diagnostics": diag}
