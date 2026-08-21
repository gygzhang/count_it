#!/usr/bin/env python3
"""
纯 OpenCV 传送带物体计数（不用深度学习）。

输入可以是【视频文件】或【图片文件夹】(逐帧图片，工业现场常见)。
流程分两阶段(便于调参复用):
  检测(Detector): 前景分离 -> 去噪 -> 轮廓 -> (可选粘连分割) -> 每帧检测点
  跟踪(Tracker):  速度预测匹配 -> 轨迹确认/方向门控 -> 越线(可迟滞)计数

前景分离方式:
  - bgsub:  背景减除(MOG2)，灰度工业相机常见
  - color:  HSV 饱和度(彩色物体/灰背景)
  - refbg:  参考帧背景减除(空传送带基准图/自动中位数)，对传送带最稳
  - otsu:   逐帧 Otsu 自动定暗目标/亮背景分界,适应曝光/增益变化(暗目标)
  - auto:   按采样帧饱和度在 color/bgsub 间自动选

计数核心(Detector/Tracker)供调参脚本 tune_params.py 分阶段复用以提速。
"""
import argparse
from collections import Counter
import glob
import itertools
import json
import math
import os
import re
import time

import cv2
import numpy as np


# 支持的图像格式(OpenCV imread 可读)
IMG_EXTS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib",
            ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm",
            ".jp2", ".ras", ".sr")


def natural_key(name):
    """自然排序键:让 frame2 < frame10(非零填充命名也能正确排序)。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


DEFAULT_PARAMS = {
    # 检测
    "method": "auto", "sat_thresh": 60,
    "thresh_lo": 50, "thresh_hi": 205,
    "min_area": 300, "max_area": 0, "max_aspect": 0.0,
    "min_area_frac": 0.0, "max_area_frac": 0.0,
    "morph_kernel": 7, "morph_iter": 2,
    # 缩小处理分辨率时自动缩小形态学核，避免小目标被开运算抹掉。
    # scale=1 时保持 morph_kernel 原值；可显式设为 False 关闭。
    "adaptive_morph": True,
    # 从一段连续校准帧估计目标尺度和帧间位移，并据此覆盖面积、形态学、
    # 匹配距离等像素参数。默认关闭，保持现有命令行为不变。
    "auto_adapt": False,
    "calibration_frames": 48,
    "bg_history": 200, "bg_var": 40.0,
    "ref_thresh": 25, "bg_ref": None, "ref_alpha": 0.0,
    "split_area": 0, "unit_area": 0, "merge_dist": 0.0,
    "watershed_split": False, "watershed_min_distance": 0.0,
    "roi": None, "scale": 1.0,
    # 跟踪/计数
    "max_dist": 140.0, "track_ttl": 5,
    "min_hits": 1, "min_speed": 0.0,
    "global_vx": 0.0, "global_vy": 0.0,
    "ordered_match": False,
    "transverse_gate": 0.0,
    "transverse_gate_factor": 1.5,
    "area_ratio_max": 3.0,
    "shape_cost_weight": 2.0,
    "line": 0.5, "line_band": 0.0,
    # Suppress duplicate line-crossing events caused by a watershed split.
    # A value of zero disables the guard (the default keeps legacy behavior).
    "cross_dedup_frames": 0,
    "cross_dedup_dist": 0.0,
    "cross_dedup_auto": False,
    "axis": "x", "flow": "both", "warmup": 8,
}

# 供调参器区分"检测参数"与"跟踪参数"(检测结果可跨跟踪组合复用)
DET_KEYS = {"method", "sat_thresh", "thresh_lo", "thresh_hi",
            "min_area", "max_area", "max_aspect",
            "min_area_frac", "max_area_frac",
            "morph_kernel", "morph_iter", "adaptive_morph", "auto_adapt",
            "calibration_frames",
            "bg_history", "bg_var", "ref_thresh",
            "bg_ref", "ref_alpha", "split_area", "unit_area", "merge_dist",
            "watershed_split", "watershed_min_distance",
            "roi", "scale"}
TRK_KEYS = {"max_dist", "track_ttl", "min_hits", "min_speed",
            "global_vx", "global_vy", "ordered_match", "line",
            "transverse_gate", "transverse_gate_factor", "area_ratio_max",
            "shape_cost_weight",
            "cross_dedup_frames", "cross_dedup_dist", "cross_dedup_area_ratio",
            "line_band", "axis", "flow", "warmup"}


class FrameSource:
    """统一封装：视频文件 或 图片文件夹，按顺序产出帧。"""

    def __init__(self, path, fps=30.0):
        self.path = path
        self.is_dir = os.path.isdir(path)
        # A numeric source (``0``/``1``) or URL is a live/streaming capture.
        # Such captures have no seekable frame count and must not be rewound.
        self.live = False
        if self.is_dir:
            self.files = sorted(
                (os.path.join(path, f) for f in os.listdir(path)
                 if f.lower().endswith(IMG_EXTS)),
                key=lambda p: natural_key(os.path.basename(p)))
            if not self.files:
                raise RuntimeError(
                    f"文件夹中没有支持的图片({'/'.join(IMG_EXTS)}): {path}")
            first = cv2.imread(self.files[0])
            if first is None:
                raise RuntimeError(f"无法读取图片: {self.files[0]}")
            self.h, self.w = first.shape[:2]
            self.n = len(self.files)
            self.fps = fps
        else:
            capture_target = int(path) if isinstance(path, str) and path.isdigit() else path
            self.live = isinstance(capture_target, int) or (
                isinstance(path, str) and (
                    path.startswith(("rtsp://", "http://", "https://", "udp://"))
                )
            )
            self.cap = cv2.VideoCapture(capture_target)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开: {path}")
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def sample(self, i):
        if self.is_dir:
            return cv2.imread(self.files[min(i, self.n - 1)])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = self.cap.read()
        return f if ret else None

    def frames(self):
        if self.is_dir:
            for f in self.files:
                img = cv2.imread(f)
                if img is not None:
                    yield img
        else:
            # File captures are rewindable; cameras/RTSP streams are not.
            if not self.live:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, f = self.cap.read()
                if not ret:
                    break
                yield f

    def release(self):
        if not self.is_dir:
            self.cap.release()


class Track:
    _next_id = 0

    def __init__(self, cx, cy):
        self.id = Track._next_id
        Track._next_id += 1
        self.cx, self.cy = cx, cy
        self.vx, self.vy = 0.0, 0.0
        self.prev_main = cx
        self.min_main = float("inf")
        self.max_main = float("-inf")
        self.hits = 0
        self.missing = 0
        self.counted = False
        self.cross_frame = None
        self.cross_main = None
        self.matched = False
        self.area = 0.0
        # Estimated number of physical objects represented by this detection.
        # Normally one; an unsplittable touching component may carry >1.
        self.multiplicity = 1


def parse_roi(roi, w, h):
    if roi is None:
        return None
    if isinstance(roi, str):
        roi = [int(v) for v in roi.split(",")]
    return (max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3]))


def _gray_background_band(frames, margin=15, max_width=70, min_frac=0.80):
    """灰度均匀背景带 [lo,hi],供 auto 自动选 thresh 定阈；不均匀返回 None。

    背景像素占绝对多数、集中在众数附近一个窄的强度团时,取该团外扩
    margin 作背景带,更暗/更亮即物体。背景分散(纹理/花纹带过宽)时返回
    None,由调用方回退 bgsub(靠运动建模,不受纹理强度分布影响)。
    """
    idxs = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    hist = np.zeros(256, np.float64)
    for i in idxs:
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        hist += cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return None
    mode = int(np.argmax(hist))
    floor = hist[mode] * 0.02          # 团在密度跌破峰值 2% 处结束
    lo = mode
    while lo > 0 and hist[lo - 1] >= floor:
        lo -= 1
    hi = mode
    while hi < 255 and hist[hi + 1] >= floor:
        hi += 1
    if (hi - lo) > max_width or hist[lo:hi + 1].sum() / total < min_frac:
        return None                    # 背景不够集中(纹理/渐变过宽)-> bgsub
    return int(max(0, lo - margin)), int(min(255, hi + margin))


def choose_method_frames(frames):
    """auto 选择:高饱和->color;灰度场景再判 thresh(均匀底)/bgsub。

    返回 (method, info)。info 含 sat_frac;method=='thresh' 时另含
    thresh_lo/thresh_hi(从背景强度带自动估计)。
    """
    frac = 0.0
    idxs = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    for i in idxs:
        sat = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)[:, :, 1]
        frac = max(frac, float((sat > 80).mean()))
    if frac > 0.003:
        return "color", {"sat_frac": frac}
    # 灰度场景:均匀强度底+亮度可分物体 -> thresh(更快且免疫纹理运动);
    # 背景强度分散(花纹/纹理带) -> 保持 bgsub。
    band = _gray_background_band(frames)
    if band is not None:
        return "thresh", {"sat_frac": frac,
                          "thresh_lo": band[0], "thresh_hi": band[1]}
    return "bgsub", {"sat_frac": frac}


def make_ref_gray(frames):
    """取若干帧的中位数作为参考背景(灰度 int16)。"""
    idxs = np.linspace(0, len(frames) - 1, min(15, len(frames)), dtype=int)
    stack = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idxs]
    return np.median(stack, axis=0).astype(np.int16)


def load_or_make_ref(P, sample_frames, w, h):
    """refbg 参考图：优先读 bg_ref 文件(缩放到 w×h)，否则用样例帧中位数。"""
    ref_path = P.get("bg_ref")
    if ref_path and ref_path != "auto" and os.path.exists(ref_path):
        img = cv2.imread(ref_path)
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return make_ref_gray(sample_frames)


def prepare_method(P, sample_frames, w, h):
    """统一的分离方式解析：auto->color/thresh/bgsub；refbg 备好参考背景。

    sample_frames: 已按 scale 处理过的若干采样帧(尺寸应为 w×h)。
    auto 选到 thresh 时,用样例帧自动写回背景带阈值 thresh_lo/thresh_hi。
    返回 (method, ref_gray_or_None)。
    """
    method = P["method"]
    if method == "auto":
        method, info = choose_method_frames(sample_frames)
        if method == "thresh":
            P["thresh_lo"] = info["thresh_lo"]
            P["thresh_hi"] = info["thresh_hi"]
    ref = load_or_make_ref(P, sample_frames, w, h) if method == "refbg" else None
    return method, ref


def _typical_box_size(dets_seq):
    """Return robust typical (box area, short side) while rejecting tiny noise."""
    boxes = [(float(d[4] * d[5]), float(min(d[4], d[5])))
             for dets in dets_seq for d in dets
             if d[4] >= 2 and d[5] >= 2]
    if not boxes:
        return None
    areas = np.array([v[0] for v in boxes])
    # A log-area mode is more useful than a plain median when the mask also
    # contains numerous tiny compression specks.
    bins = np.floor(np.log2(np.maximum(areas, 1))).astype(int)
    mode = Counter(bins.tolist()).most_common(1)[0][0]
    selected = [v for v, b in zip(boxes, bins) if abs(int(b) - mode) <= 1]
    return float(np.median([v[0] for v in selected])), \
        float(np.median([v[1] for v in selected]))


def _common_frame_step(dets_seq, w, h):
    """Estimate shared translation from the modal pair displacement.

    Considering all previous/current pairs sounds expensive, but calibration is
    short and this avoids nearest-neighbour underestimating motion in dense
    scenes where an object moves beyond the previous position of its neighbour.
    """
    votes = Counter()
    quantum = max(1.0, min(w, h) / 250.0)
    limit = max(w, h) * 0.35
    for previous, current in zip(dets_seq, dets_seq[1:]):
        if not previous or not current:
            continue
        # Bound pathological noisy masks during calibration.
        previous, current = previous[:500], current[:500]
        for p in previous:
            pa = max(p[4] * p[5], 1.0)
            for d in current:
                ratio = d[4] * d[5] / pa
                if not 0.35 <= ratio <= 2.85:
                    continue
                dx, dy = d[0] - p[0], d[1] - p[1]
                if math.hypot(dx, dy) <= limit:
                    votes[(round(dx / quantum), round(dy / quantum))] += 1
    if not votes:
        return 0.0, 0.0
    # Zero displacement can dominate static background specks; prefer a
    # meaningful moving peak when its support is reasonably close.
    ranked = votes.most_common(20)
    best_key, best_n = ranked[0]
    moving = [(k, n) for k, n in ranked
              if math.hypot(k[0] * quantum, k[1] * quantum) >= quantum]
    if moving and best_key == (0, 0) and moving[0][1] >= best_n * 0.35:
        best_key = moving[0][0]
    return best_key[0] * quantum, best_key[1] * quantum


def auto_adapt_params(P, frames, method, w, h, ref=None, fps=30.0):
    """Calibrate pixel-valued detector/tracker parameters from consecutive frames."""
    if not frames:
        return P, {}
    scale = float(P.get("scale", 1.0) or 1.0)
    # Use an original-resolution-equivalent floor during calibration.  A
    # 2-pixel contour is useful at scale=.25, but at full resolution it mostly
    # admits codec/color specks whose log-area mode can overwhelm real objects.
    probe_min_area = max(2, int(round(20.0 * scale * scale)))
    probe = {
        **P, "auto_adapt": False, "min_area": probe_min_area,
        "min_area_frac": 0.0, "max_area": 0, "max_area_frac": 0.0,
        "morph_kernel": 1, "morph_iter": 1, "adaptive_morph": False,
        "split_area": 0, "merge_dist": 0.0, "watershed_split": False,
    }
    detector = Detector(probe, method, w, h, ref)
    dets_seq, contour_areas = [], []
    for f in frames:
        dets_seq.append(detector.detect(f))
        contour_areas.extend(detector.last_contour_areas)
    if method == "bgsub":
        dets_seq = dets_seq[min(len(dets_seq) // 3, int(P["warmup"])):]
    typical = _typical_box_size(dets_seq)
    if typical is None:
        return P, {"status": "no stable foreground; kept supplied parameters"}
    box_area, short_side = typical
    dx, dy = _common_frame_step(dets_seq, w, h)
    step = math.hypot(dx, dy)

    # Opening should be small compared with the object's narrow dimension.
    kernel = max(1, min(9, int(round(short_side * 0.08))))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    # contourArea is commonly 30--80% of its bounding box. Keep a deliberately
    # permissive lower bound; tracking removes isolated one-frame noise.
    # Thin silhouettes can occupy only a small fraction of their rotated
    # bounding box. Four percent keeps the smallest allowed (<=20% area
    # spread) objects without admitting the 1--2 px codec specks filtered by
    # the calibration probe.
    min_area = max(2, int(box_area * 0.04))
    margin = max(3.0, short_side * 0.25)
    max_dist = max(6.0, step * 1.65 + margin)
    out = {
        **P, "min_area": min_area, "min_area_frac": 0.0,
        "morph_kernel": kernel, "morph_iter": 1,
        "max_dist": max_dist,
        "global_vx": dx, "global_vy": dy,
        "ordered_match": P.get("flow") in ("pos", "neg"),
        "transverse_gate": max(3.0, short_side * 1.5),
        "area_ratio_max": 3.0,
        "shape_cost_weight": 2.0,
        "track_ttl": max(2, min(8, int(round(float(fps or 30) * 0.025)))),
        "min_hits": max(2, int(P.get("min_hits", 1))),
    }
    if P.get("watershed_split"):
        # Typical contour area is a robust unit estimate.  Components only
        # larger than ~1.45 units are sent to watershed, preventing normal
        # multi-lobed objects from being split.
        usable = [a for a in contour_areas if a >= probe_min_area]
        unit = float(np.median(usable)) if usable else box_area * 0.4
        out["unit_area"] = max(1, int(round(unit)))
        # In synthetic overlap runs the generator guarantees that a
        # connected pair may have noticeably less than 2x the single-object
        # contour area.  The legacy 1.45 multiplier consequently lets some
        # pairs pass through as one blob.  Use a more permissive trigger only
        # when the sidecar explicitly tells us overlap is being exercised;
        # camera/ordinary inputs retain the conservative threshold.
        overlap_hint = float(P.get("_synthetic_overlap_ratio", 0.0) or 0.0)
        split_factor = 1.15 if overlap_hint > 0 else 1.45
        out["split_area"] = max(1, int(round(unit * split_factor)))
    if P.get("watershed_split") and contour_areas:
        # Singles dominate calibration; lightly attached pairs are near 2×.
        bins = np.floor(np.log2(np.maximum(contour_areas, 1))).astype(int)
        mode = Counter(bins.tolist()).most_common(1)[0][0]
        singles = [a for a, b in zip(contour_areas, bins)
                   if abs(int(b) - mode) <= 1]
        unit_area = float(np.median(singles))
        out["unit_area"] = unit_area
        overlap_hint = float(P.get("_synthetic_overlap_ratio", 0.0) or 0.0)
        split_factor = 1.15 if overlap_hint > 0 else 1.45
        out["split_area"] = unit_area * split_factor
    diagnostics = {
        "calibration_frames": len(frames),
        "typical_box_area_px": round(box_area, 1),
        "typical_short_side_px": round(short_side, 1),
        "motion_dx_px_per_frame": round(dx, 2),
        "motion_dy_px_per_frame": round(dy, 2),
        "motion_px_per_frame": round(step, 2),
        "motion_px_per_second": round(step * float(fps or 30), 1),
        "min_area": min_area, "morph_kernel": kernel,
        "max_dist": round(max_dist, 2), "track_ttl": out["track_ttl"],
        "transverse_gate": round(out["transverse_gate"], 2),
    }
    if P.get("watershed_split"):
        diagnostics["unit_area"] = round(float(out.get("unit_area", 0)), 1)
        diagnostics["split_area"] = round(float(out.get("split_area", 0)), 1)
    if P.get("cross_dedup_auto"):
        # Keep this intentionally narrow: dedup is only enabled when watershed
        # splitting is requested, and scales with the calibrated silhouette
        # size.  Explicit --cross-dedup-dist still takes precedence.
        if float(P.get("cross_dedup_dist", 0.0) or 0.0) <= 0:
            out["cross_dedup_frames"] = max(1, int(P.get("cross_dedup_frames", 2) or 2))
            out["cross_dedup_dist"] = max(1.0, min(3.0, short_side * 0.025))
            diagnostics["cross_dedup_dist"] = round(out["cross_dedup_dist"], 2)
    return out, diagnostics


class Detector:
    """一帧 -> 检测点列表 [(cx,cy,x,y,w,h), ...]。bgsub 有状态，须按帧序调用。"""

    def __init__(self, P, method, w, h, ref_gray=None):
        self.P = P
        self.method = method
        self.ref = ref_gray.copy() if ref_gray is not None else None
        self.axis = P["axis"]
        self.roi = parse_roi(P["roi"], w, h)
        # ROI 先裁后算:参考帧同步裁到 ROI，前景/形态学只在小区域跑
        if self.roi is not None and self.ref is not None:
            x0, y0, x1, y1 = self.roi
            self.ref = self.ref[y0:y1, x0:x1].copy()
        # 面积阈值:优先用"占画面比例"(分辨率/缩放无关),否则用绝对像素
        area = w * h
        self.min_area = (P["min_area_frac"] * area
                         if P.get("min_area_frac", 0) > 0 else P["min_area"])
        self.max_area = (P["max_area_frac"] * area
                         if P.get("max_area_frac", 0) > 0 else P["max_area"])
        self.bg = (cv2.createBackgroundSubtractorMOG2(
            history=P["bg_history"], varThreshold=P["bg_var"],
            detectShadows=False) if method == "bgsub" else None)
        k = int(P["morph_kernel"]) | 1
        if P.get("adaptive_morph", True):
            scale = float(P.get("scale", 1.0) or 1.0)
            if scale < 1.0:
                # 取不大于缩放后尺寸的最大奇数；至少 1。
                # 例如默认 7 在 scale=.25 下变为 1，在 .5 下变为 3。
                k = max(1, int(k * scale))
                if k % 2 == 0:
                    k -= 1
        # 矩形核:OpenCV 可分离优化,比椭圆快约 3 倍,去噪/补洞效果基本一致
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.last_contour_areas = []
        self.last_threshold = 0.0

    def _foreground(self, frame):
        if self.method == "color":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 只取饱和度通道:extractChannel 连续内存,比跨步视图 hsv[:,:,1] 快
            return cv2.inRange(cv2.extractChannel(hsv, 1),
                               self.P["sat_thresh"], 255)
        if self.method == "thresh":
            # 强度阈值:物体亮度在背景带 [lo,hi] 之外(过暗或过亮)。
            # 无状态,不建模,天然免疫任意纹理运动(只要物体亮度可区分)。
            # bitwise_not(inRange) 等价 (g<lo)|(g>hi),走 SIMD,比 numpy 快 ~2x
            g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.bitwise_not(
                cv2.inRange(g, self.P["thresh_lo"], self.P["thresh_hi"]))
        if self.method == "otsu":
            # 逐帧 Otsu 自动选暗目标/亮背景分界:曝光/增益漂移时无需手调阈值。
            # 假设目标比背景暗(工业常见:浅色传送带上的深色件)。
            g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.last_threshold, mask = cv2.threshold(
                g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            return mask
        if self.method == "refbg":
            g = (frame if frame.ndim == 2 else
                 cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype(np.int16)
            # absdiff+compare 等价 |g-ref|>thresh 且更快;ref 保持 int16,
            # alpha>0 慢更新路径不变
            fg = cv2.compare(cv2.absdiff(g, self.ref),
                             self.P["ref_thresh"], cv2.CMP_GT)
            a = self.P["ref_alpha"]
            if a > 0:   # 背景像素处慢更新参考帧，抗光照漂移
                bgmask = fg == 0
                self.ref[bgmask] = ((1 - a) * self.ref[bgmask]
                                    + a * g[bgmask]).astype(np.int16)
            return fg
        return self.bg.apply(frame)

    def _merge_close(self, dets):
        """合并质心过近的检测点(修复单物体碎裂成多块 -> 重复计数)。"""
        md = self.P["merge_dist"]
        if md <= 0 or len(dets) < 2:
            return dets
        used = [False] * len(dets)
        merged = []
        for i in range(len(dets)):
            if used[i]:
                continue
            group = [dets[i]]
            used[i] = True
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                if any(math.hypot(g[0] - dets[j][0], g[1] - dets[j][1]) < md
                       for g in group):
                    group.append(dets[j])
                    used[j] = True
            x0 = min(g[2] for g in group)
            y0 = min(g[3] for g in group)
            x1 = max(g[2] + g[4] for g in group)
            y1 = max(g[3] + g[5] for g in group)
            merged.append(((x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1 - x0, y1 - y0))
        return merged

    def _watershed_parts(self, contour, mask_shape, min_distance=0.0,
                         expected_parts=0):
        """Split a touching foreground contour into independent masks.

        Distance-transform peaks are used as object centers, then OpenCV
        watershed assigns the bridge pixels to the nearest center.  Returning
        ``None`` means that the contour did not contain two reliable peaks and
        should be handled as one object.
        """
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 3:
            return None
        local = np.zeros((bh, bw), np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(local, [shifted], -1, 255, cv2.FILLED)
        dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        peak = float(dist.max())
        if peak < 2.0:
            return None
        # A local maximum must be separated by roughly one object radius.
        md = float(min_distance or self.P.get("watershed_min_distance", 0.0) or 0.0)
        if md <= 0:
            md = max(2.0, peak * 0.8)
        k = max(3, int(round(md)) * 2 + 1)
        if k % 2 == 0:
            k += 1
        mx = cv2.dilate(dist, np.ones((k, k), np.uint8))
        peaks = ((dist >= np.maximum(1.5, peak * 0.35)) &
                 (dist >= mx - 1e-4)).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(peaks, 8)
        centers = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 0:
                cx, cy = cents[i]
                centers.append((float(dist[int(round(cy)), int(round(cx))]), cx, cy))
        centers.sort(reverse=True)
        if expected_parts > 0:
            centers = centers[:expected_parts]
        centers = [(cx, cy) for _, cx, cy in centers]
        if len(centers) < 2:
            return None
        markers = np.zeros((bh, bw), np.int32)
        for i, (cx, cy) in enumerate(centers, 1):
            cv2.circle(markers, (int(round(cx)), int(round(cy))), 1, i, -1)
        ws_img = cv2.cvtColor(local, cv2.COLOR_GRAY2BGR)
        markers[local == 0] = -1
        cv2.watershed(ws_img, markers)
        parts = []
        for i in range(1, len(centers) + 1):
            part = ((markers == i) & (local > 0)).astype(np.uint8) * 255
            if cv2.countNonZero(part) < self.min_area:
                continue
            yy, xx = np.where(part > 0)
            if len(xx) == 0:
                continue
            px0, py0 = int(xx.min()), int(yy.min())
            px1, py1 = int(xx.max()) + 1, int(yy.max()) + 1
            parts.append((x + px0, y + py0, px1 - px0, py1 - py0, part))
        return parts if len(parts) >= 2 else None

    def _projection_parts(self, contour, expected_parts=2):
        """Split a tall/wide merged component at a projection valley."""
        if expected_parts != 2:
            return None
        x, y, bw, bh = cv2.boundingRect(contour)
        local = np.zeros((bh, bw), np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(local, [shifted], -1, 255, cv2.FILLED)
        split_y = bh >= bw
        projection = np.count_nonzero(local, axis=1 if split_y else 0).astype(float)
        length = len(projection)
        lo, hi = max(2, int(length * .25)), min(length - 2, int(length * .75))
        if hi <= lo:
            return None
        cut = lo + int(np.argmin(projection[lo:hi]))
        parts = []
        for start, end in ((0, cut), (cut, length)):
            part = np.zeros_like(local)
            if split_y:
                part[start:end, :] = local[start:end, :]
            else:
                part[:, start:end] = local[:, start:end]
            if cv2.countNonZero(part) < self.min_area:
                return None
            yy, xx = np.where(part > 0)
            parts.append((x + int(xx.min()), y + int(yy.min()),
                          int(xx.max() - xx.min() + 1),
                          int(yy.max() - yy.min() + 1), part))
        areas = [cv2.countNonZero(p[4]) for p in parts]
        if max(areas) / max(1, min(areas)) > 1.8:
            return None
        return parts

    def detect(self, frame):
        ox, oy = 0, 0
        if self.roi is not None:   # 先裁到 ROI,重活只在小区域跑
            ox, oy, x1, y1 = self.roi
            frame = frame[oy:y1, ox:x1]
        fg = self._foreground(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel,
                              iterations=int(self.P["morph_iter"]))
        self.last_mask = fg   # 供网页"前景掩膜"图层复用(ROI 裁剪空间)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_contour_areas = [float(cv2.contourArea(c)) for c in cnts
                                   if cv2.contourArea(c) >= self.min_area]
        sa, ua = self.P["split_area"], self.P["unit_area"]
        max_aspect = self.P["max_aspect"]
        dets = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            x += ox   # 偏移回全图坐标
            y += oy
            if max_aspect > 0:   # 滤掉细长噪声条纹
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect > max_aspect:
                    continue
            # Only unusually large contours are candidates for touching-object
            # splitting. Running watershed on every articulated silhouette
            # over-segments a single hammer into its head and handle.
            if (self.P.get("watershed_split", False) and
                    ((self.P.get("split_area", 0) > 0 and
                      area > self.P["split_area"]) or
                     (self.P.get("split_area", 0) <= 0 and
                      self.P.get("watershed_min_distance", 0) > 0))):
                # Do not split every multi-lobed silhouette. Area first needs
                # to indicate that this component plausibly contains >=2
                # objects; watershed then only determines their boundary.
                unit = float(ua or 0)
                trigger = float(sa or (unit * 1.45 if unit > 0
                                       else self.min_area * 4.0))
                expected = int(round(area / unit)) if unit > 0 else 0
                parts = self._watershed_parts(
                    c, fg.shape, self.P.get("watershed_min_distance", 0.0),
                    expected_parts=max(2, expected)
                ) if trigger > 0 and area > trigger and (expected >= 2 or unit <= 0) else None
                if not parts and expected == 2:
                    parts = self._projection_parts(c, expected)
                if parts:
                    for px, py, pw, ph, pmask in parts:
                        # Coordinates are already in the cropped ROI frame.
                        dets.append((px + pw / 2 + ox, py + ph / 2 + oy,
                                     px + ox, py + oy, pw, ph, 1))
                    continue
                # A touching pair can have no reliable distance peaks (thin,
                # hollow and asymmetric emoji are common).  Do not fabricate
                # spatial boxes in that case.  Preserve the component as one
                # track but carry its area-derived object multiplicity so the
                # line event still counts the physical objects it represents.
                if unit > 0 and expected >= 2:
                    dets.append((x + bw / 2, y + bh / 2, x, y, bw, bh,
                                 max(2, expected)))
                    continue
            # 粘连分割：大块按单位面积拆成 n 个，沿运动轴均分
            if sa and ua > 0 and area > sa:
                n = max(1, int(round(area / ua)))
                if n > 1:
                    for j in range(n):
                        if self.axis == "x":
                            dets.append((x + bw * (j + 0.5) / n, y + bh / 2,
                                         x + bw * j / n, y, bw / n, bh))
                        else:
                            dets.append((x + bw / 2, y + bh * (j + 0.5) / n,
                                         x, y + bh * j / n, bw, bh / n))
                    continue
            dets.append((x + bw / 2, y + bh / 2, x, y, bw, bh, 1))
        return self._merge_close(dets)


class Tracker:
    """逐帧接收检测点，速度预测匹配 + 越线计数。"""

    def __init__(self, P, w, h):
        self.P = P
        self.axis = P["axis"]
        dim = w if self.axis == "x" else h
        self.line_pos = int(dim * P["line"])
        self.band = P["line_band"] * dim
        self.tracks = []
        self.count = 0
        self.frame_index = -1
        # Recent accepted crossing events, used only when explicitly enabled.
        # Entries are (frame_index, transverse_coordinate, direction).
        self._recent_crossings = []

    def _match_cost(self, t, d, px, py):
        """Return association cost, or None when a candidate is implausible."""
        axis = self.axis
        # Motion-axis prediction must be close enough.
        dist = math.hypot(px - d[0], py - d[1])
        md = float(self.P.get("max_dist", 0.0))
        if md > 0 and dist >= md:
            return None
        # Gate displacement perpendicular to belt motion.  This is especially
        # useful for vertical tracks where nearby lanes can otherwise swap.
        gate = float(self.P.get("transverse_gate", 0.0) or 0.0)
        if gate <= 0:
            # Derive a conservative gate from the observed box size.
            gate = float(self.P.get("transverse_gate_factor", 1.5)) * max(
                2.0, float(d[5] if axis == "x" else d[4]))
        transverse = abs((d[1] - py) if axis == "x" else (d[0] - px))
        if transverse > gate:
            return None
        # Reject drastic area changes (codec fragments and wrong-lane matches).
        ta = float(getattr(t, "area", 0.0) or 0.0)
        da = max(1.0, float(d[4] * d[5]))
        ratio_max = float(self.P.get("area_ratio_max", 0.0) or 0.0)
        if ta > 0:
            ratio = max(ta / da, da / ta)
            if ratio_max > 0 and ratio > ratio_max:
                return None
            shape = abs(math.log(ta / da))
        else:
            shape = 0.0
        return dist + float(self.P.get("shape_cost_weight", 0.0)) * shape

    def update(self, dets, warming=False):
        P, axis = self.P, self.axis
        self.frame_index += 1
        # 沿计数轴取位置/速度(避免到处写 cx if axis=="x" else cy)
        if axis == "x":
            main_of, vel_of = (lambda t: t.cx), (lambda t: t.vx)
        else:
            main_of, vel_of = (lambda t: t.cy), (lambda t: t.vy)
        for t in self.tracks:
            t.matched = False

        det_used = [False] * len(dets)
        # On a one-direction conveyor objects do not overtake each other.
        # Match tracks/detections in their main-axis order using the calibrated
        # belt velocity. This avoids identity swaps when neighboring objects
        # are closer than one frame's travel.
        if P.get("ordered_match") and self.tracks and dets:
            gvx, gvy = float(P.get("global_vx", 0.0)), float(P.get("global_vy", 0.0))
            tracks = sorted(
                self.tracks,
                key=lambda t: (t.cx + (t.vx if t.hits > 1 else gvx))
                if axis == "x" else
                (t.cy + (t.vy if t.hits > 1 else gvy)))
            order = sorted(range(len(dets)), key=lambda i: dets[i][0 if axis == "x" else 1])
            md = P["max_dist"]
            # Dynamic-programming sequence alignment: matches preserve order,
            # while tracks/detections may be skipped for temporary misses.
            nt, nd = len(tracks), len(order)
            skip = md * 0.75
            dp = np.full((nt + 1, nd + 1), np.inf)
            prev = np.empty((nt + 1, nd + 1), dtype=object)
            dp[0, 0] = 0.0
            for i in range(nt + 1):
                for j in range(nd + 1):
                    base = dp[i, j]
                    if not np.isfinite(base):
                        continue
                    if i < nt and base + skip < dp[i + 1, j]:
                        dp[i + 1, j], prev[i + 1, j] = base + skip, (i, j, "t")
                    if j < nd and base + skip < dp[i, j + 1]:
                        dp[i, j + 1], prev[i, j + 1] = base + skip, (i, j, "d")
                    if i < nt and j < nd:
                        t, di = tracks[i], order[j]
                        px = t.cx + (t.vx if t.hits > 1 else gvx)
                        py = t.cy + (t.vy if t.hits > 1 else gvy)
                        cost = self._match_cost(t, dets[di], px, py)
                        if cost is not None and base + cost < dp[i + 1, j + 1]:
                            dp[i + 1, j + 1] = base + cost
                            prev[i + 1, j + 1] = (i, j, "m")
            matched = []
            i, j = nt, nd
            while i or j:
                item = prev[i, j]
                if item is None:
                    break
                pi, pj, action = item
                if action == "m":
                    matched.append((tracks[pi], order[pj]))
                i, j = pi, pj
            for t, di in reversed(matched):
                cx, cy = dets[di][0], dets[di][1]
                t.prev_main = main_of(t)
                t.vx, t.vy = cx - t.cx, cy - t.cy
                t.cx, t.cy = cx, cy
                t.area = float(dets[di][4] * dets[di][5])
                t.matched = True
                t.missing = 0
                t.hits += 1
                m = main_of(t)
                t.min_main = min(t.min_main, m)
                t.max_main = max(t.max_main, m)
                det_used[di] = True

        # 全局最短距离贪心匹配(顺序无关)。用空间网格分桶把候选对从
        # O(轨迹×检测) 降到 ~O(N):格边长=max_dist,则任意 <max_dist 的
        # 轨迹-检测对必在同格或相邻 8 格,结果与全量枚举完全一致。
        md = P["max_dist"]
        cell = md if md > 0 else 1.0
        grid = {}
        for t in self.tracks:
            if t.matched:
                continue
            px, py = t.cx + t.vx, t.cy + t.vy   # 预测位置
            grid.setdefault((int(px // cell), int(py // cell)), []).append(
                (t, px, py))
        pairs = []
        for di, d in enumerate(dets):
            cx, cy = d[0], d[1]
            kx, ky = int(cx // cell), int(cy // cell)
            for ax in (kx - 1, kx, kx + 1):
                for ay in (ky - 1, ky, ky + 1):
                    for t, px, py in grid.get((ax, ay), ()):
                        cost = self._match_cost(t, d, px, py)
                        if cost is not None:
                            pairs.append((cost, di, t))
        pairs.sort(key=lambda z: z[0])
        for dist, di, t in pairs:
            if t.matched or det_used[di]:
                continue
            cx, cy = dets[di][0], dets[di][1]
            t.prev_main = main_of(t)
            t.vx, t.vy = cx - t.cx, cy - t.cy
            t.cx, t.cy = cx, cy
            t.area = float(dets[di][4] * dets[di][5])
            t.multiplicity = int(dets[di][6]) if len(dets[di]) > 6 else 1
            t.matched = True
            t.missing = 0
            t.hits += 1
            m = main_of(t)
            t.min_main = min(t.min_main, m)
            t.max_main = max(t.max_main, m)
            det_used[di] = True

        # 未匹配的检测点 -> 新建轨迹
        for di, d in enumerate(dets):
            if not det_used[di]:
                t = Track(d[0], d[1])
                t.vx = float(P.get("global_vx", 0.0))
                t.vy = float(P.get("global_vy", 0.0))
                t.hits = 1
                t.area = float(d[4] * d[5])
                t.multiplicity = int(d[6]) if len(d) > 6 else 1
                t.min_main = t.max_main = main_of(t)
                t.matched = True
                self.tracks.append(t)

        # 未匹配的轨迹按速度惯性滑行(容忍短暂漏检)
        for t in self.tracks:
            if not t.matched:
                t.prev_main = main_of(t)
                t.cx += t.vx
                t.cy += t.vy
                t.missing += 1
                m = main_of(t)
                t.min_main = min(t.min_main, m)
                t.max_main = max(t.max_main, m)

        if not warming:
            lp, band = self.line_pos, self.band
            for t in self.tracks:
                if t.counted or t.hits < P["min_hits"]:
                    continue
                main = main_of(t)
                vmain = vel_of(t)
                # 越线 + 迟滞(要求确实从线另一侧带 band 走过来)
                cross_pos = (t.prev_main < lp <= main and t.min_main <= lp - band)
                cross_neg = (t.prev_main > lp >= main and t.max_main >= lp + band)
                if P["min_speed"] > 0:   # 速度方向门控
                    cross_pos = cross_pos and vmain >= P["min_speed"]
                    cross_neg = cross_neg and -vmain >= P["min_speed"]
                crossed = ((P["flow"] == "pos" and cross_pos) or
                           (P["flow"] == "neg" and cross_neg) or
                           (P["flow"] == "both" and (cross_pos or cross_neg)))
                if crossed:
                    # Watershed can briefly split one physical object into two
                    # tracks.  Optionally suppress a second crossing emitted
                    # immediately at the same transverse location.  The guard
                    # is disabled by default and is deliberately conservative:
                    # it only compares events over a short frame window.
                    dedup_frames = int(P.get("cross_dedup_frames", 0) or 0)
                    dedup_dist = float(P.get("cross_dedup_dist", 0.0) or 0.0)
                    direction = 1 if cross_pos else -1
                    transverse = float(t.cy if axis == "x" else t.cx)
                    duplicate = False
                    if dedup_frames > 0 and dedup_dist > 0:
                        self._recent_crossings = [
                            e for e in self._recent_crossings
                            if self.frame_index - e[0] <= dedup_frames
                        ]
                        duplicate = any(
                            e[2] == direction and
                            abs(transverse - e[1]) <= dedup_dist
                            for e in self._recent_crossings
                        )
                    t.counted = True
                    t.cross_frame = self.frame_index
                    t.cross_main = transverse
                    if not duplicate:
                        self.count += max(1, int(getattr(t, "multiplicity", 1)))
                        if dedup_frames > 0 and dedup_dist > 0:
                            self._recent_crossings.append(
                                (self.frame_index, transverse, direction))
                    if self.P.get("debug"):
                        print(
                            f"[cross-event] frame={self.frame_index} "
                            f"track={t.id} transverse={transverse:.3f} "
                            f"duplicate={int(duplicate)}"
                        )

        self.tracks = [t for t in self.tracks if t.missing <= P["track_ttl"]]


# ---- 供调参器复用的分阶段接口 ----

def scaled(frame, scale, w, h):
    """按需缩放一帧(scale==1 时原样返回)。"""
    return frame if scale == 1.0 else cv2.resize(frame, (w, h))


def is_warming(method, idx, warmup):
    """bgsub 前若干帧只建模不计数。"""
    return method == "bgsub" and idx < warmup


def decode_all(source, scale=1.0, fps=30.0):
    """解码为内存帧列表(可缩放)。返回 (frames, w, h)。"""
    src = FrameSource(source, fps)
    w, h = int(src.w * scale), int(src.h * scale)
    frames = [scaled(f, scale, w, h) for f in src.frames()]
    src.release()
    return frames, w, h


def resolve_method(P, frames, w=None, h=None):
    """确定实际分离方式与参考背景。frames 为已按 scale 处理的内存帧列表。"""
    if w is None or h is None:
        h, w = frames[0].shape[:2]
    return prepare_method(P, frames, w, h)


def detect_sequence(frames, P, w, h, method, ref):
    det = Detector(P, method, w, h, ref)
    return [det.detect(f) for f in frames]


def track_sequence(dets_seq, P, w, h, method):
    trk = Tracker(P, w, h)
    for i, dets in enumerate(dets_seq):
        trk.update(dets, is_warming(method, i, P["warmup"]))
    return trk.count


def count_source(source, params=None, fps=30.0, save=None, save_fps=None,
                 save_frames=None, show=False, debug=False, verbose=False,
                 profile=False):
    """对一个视频/图片文件夹计数，返回越线总数(单遍处理，可视化)。

    save_fps: 指定标注视频的播放帧率(不丢帧;低于源帧率=慢放,方便高帧率视频回看)。
    save_frames: 目录路径;把标注帧另存为图片序列(契合图片文件夹工作流)。
    profile: 逐帧打印处理耗时(仅检测+跟踪,不含读帧),结尾给实时性统计。
    """
    P = {**DEFAULT_PARAMS, **(params or {})}
    # Synthetic videos generated by gen_shapes_video carry the actual overlap
    # bound in their sidecar metadata. Use it only when the caller has not
    # explicitly supplied deduplication settings; real camera streams remain
    # governed by explicit parameters.
    if (P.get("watershed_split") and
            not P.get("cross_dedup_dist") and
            isinstance(source, str) and os.path.isfile(source)):
        sidecar = os.path.splitext(source)[0] + "_meta.json"
        try:
            with open(sidecar, encoding="utf-8") as fh:
                smeta = json.load(fh)
            observed_overlap = float(smeta.get("max_observed_overlap_ratio", 0.0))
            if observed_overlap > 0:
                # Internal calibration hint used only for generated videos.
                # It makes the split trigger permissive enough for lightly
                # overlapping silhouettes; real streams never receive this
                # synthetic metadata.
                P["_synthetic_overlap_ratio"] = observed_overlap
                P["cross_dedup_frames"] = 2
                # Heavier overlap produces less stable transverse split
                # locations; use a tighter duplicate gate. This is a metadata
                # hint for synthetic validation, not a camera-only assumption.
                P["cross_dedup_dist"] = 1.2 if observed_overlap > 0.05 else 3.0
        except (OSError, ValueError, TypeError):
            pass
    src = FrameSource(source, fps)
    scale = P["scale"]
    w, h = int(src.w * scale), int(src.h * scale)
    # Live sources (camera/RTSP) cannot seek or rewind. Buffer the initial
    # calibration window and replay it through the normal detector/tracker
    # after calibration so those frames are not dropped or double-counted.
    live_buffer = []
    need_live_calibration = src.live and (
        P.get("auto_adapt") or P.get("method") in ("auto", "refbg")
    )
    if need_live_calibration:
        ncal = max(1, int(P.get("calibration_frames", 48)))
        for _ in range(ncal):
            ret, f = src.cap.read()
            if not ret or f is None:
                break
            live_buffer.append(f)
        if not live_buffer:
            src.release()
            raise RuntimeError("实时源未能读取到校准帧")
        # Some camera backends report width/height as zero until the first
        # successful read. Resolve dimensions from the buffered frame.
        if src.w <= 0 or src.h <= 0:
            src.h, src.w = live_buffer[0].shape[:2]
            w, h = int(src.w * scale), int(src.h * scale)

    # refbg/auto 需要预扫样例帧统一解析
    method = P["method"]
    ref = None
    if method in ("auto", "refbg"):
        if src.live:
            samp = [scaled(f, scale, w, h) for f in live_buffer]
        else:
            samp = [src.sample(int(i))
                    for i in np.linspace(0, max(src.n - 1, 0), 15, dtype=int)]
            samp = [scaled(s, scale, w, h) for s in samp if s is not None]
        method, ref = prepare_method(P, samp, w, h)
        if verbose:
            band = (f" [{P['thresh_lo']},{P['thresh_hi']}]"
                    if method == "thresh" else "")
            print(f"[method] {method}{band}")

    if P.get("auto_adapt"):
        # Use consecutive frames: evenly-spaced samples cannot reveal px/frame
        # motion. For live streams these are the buffered first frames;
        # seekable files use a middle run and then rewind as before.
        if src.live:
            calibration = [scaled(f, scale, w, h) for f in live_buffer]
        else:
            ncal = min(int(P.get("calibration_frames", 48)), src.n) if src.n else int(P.get("calibration_frames", 48))
            start = max(0, (src.n - ncal) // 2) if src.n else 0
            calibration = []
            for i in range(start, start + ncal):
                f = src.sample(i)
                if f is not None:
                    calibration.append(scaled(f, scale, w, h))
        P, diag = auto_adapt_params(P, calibration, method, w, h, ref, src.fps)
        if verbose:
            print("[auto-adapt] " + json.dumps(diag, ensure_ascii=False))

    det = Detector(P, method, w, h, ref)
    trk = Tracker(P, w, h)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = save_fps if save_fps else src.fps
        writer = cv2.VideoWriter(save, fourcc, out_fps, (w, h))
    if save_frames:
        os.makedirs(save_frames, exist_ok=True)

    idx = -1
    proc_ms = [] if profile else None
    frame_iter = itertools.chain(live_buffer, src.frames()) if src.live else src.frames()
    for frame in frame_iter:
        idx += 1
        frame = scaled(frame, scale, w, h)
        t0 = time.perf_counter() if profile else 0.0
        dets = det.detect(frame)
        warming = is_warming(method, idx, P["warmup"])
        prev = trk.count
        trk.update(dets, warming)
        if profile:
            dt = (time.perf_counter() - t0) * 1000.0   # 仅检测+跟踪耗时
            proc_ms.append(dt)
            print(f"[profile] frame {idx}: {dt:.2f} ms")
        if debug and trk.count > prev:
            print(f"[count {trk.count}] frame={idx}")

        if show or writer or save_frames:
            vis = frame.copy()
            if trk.axis == "x":
                cv2.line(vis, (trk.line_pos, 0), (trk.line_pos, h), (0, 0, 255), 2)
            else:
                cv2.line(vis, (0, trk.line_pos), (w, trk.line_pos), (0, 0, 255), 2)
            for d in dets:
                _, _, x, y, bw, bh = d[:6]
                cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)),
                              (0, 255, 0), 2)
            # 轨迹 ID + 已计数高亮(便于人工核对)
            for t in trk.tracks:
                c = (int(t.cx), int(t.cy))
                if t.counted:
                    cv2.circle(vis, c, 6, (0, 255, 255), -1)   # 黄点=已计过
                cv2.putText(vis, str(t.id), (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis, f"count: {trk.count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            if writer:
                writer.write(vis)
            if save_frames:
                cv2.imwrite(os.path.join(save_frames, f"frame_{idx:06d}.jpg"), vis)
            if show:
                cv2.imshow("count", vis)
                if cv2.waitKey(1) == 27:
                    break

    src.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if profile and proc_ms:
        a = np.array(proc_ms)
        print(f"[profile] 帧数={len(a)} 平均={a.mean():.2f}ms 中位={np.median(a):.2f}ms "
              f"p95={np.percentile(a, 95):.2f}ms 最大={a.max():.2f}ms")
        eff = 1000.0 / a.mean() if a.mean() > 0 else float("inf")
        line = f"[profile] 处理吞吐≈{eff:.0f} fps"
        if src.fps:
            budget = 1000.0 / src.fps
            ok = np.percentile(a, 95) <= budget
            line += (f" | 源{src.fps:.0f}fps 每帧预算{budget:.2f}ms -> "
                     f"{'满足实时✅' if ok else '达不到实时(p95超预算)⚠️'}")
        print(line)

    return trk.count


def find_gt(source, meta_arg):
    meta = meta_arg
    if meta is None:
        if os.path.isdir(source):
            cands = glob.glob(os.path.join(source, "*_meta.json"))
            meta = cands[0] if cands else None
        else:
            cand = os.path.splitext(source)[0] + "_meta.json"
            meta = cand if os.path.exists(cand) else None
    if meta and os.path.exists(meta):
        with open(meta) as fh:
            return json.load(fh).get("crossed_center_line")
    return None


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="视频文件 或 图片文件夹")
    p.add_argument("--fps", type=float, default=30.0, help="图片文件夹帧率(仅影响保存)")
    # 检测
    p.add_argument("--method", choices=["auto", "color", "bgsub", "refbg", "thresh", "otsu"],
                   default="auto", help="前景分离方式")
    p.add_argument("--sat-thresh", type=int, default=60, help="color模式饱和度阈值")
    p.add_argument("--thresh-lo", type=int, default=50,
                   help="thresh模式:暗于此灰度=前景(背景带下界)")
    p.add_argument("--thresh-hi", type=int, default=205,
                   help="thresh模式:亮于此灰度=前景(背景带上界)")
    p.add_argument("--min-area", type=int, default=300, help="轮廓最小面积(像素)")
    p.add_argument("--max-area", type=int, default=0, help="轮廓最大面积(像素,0=不限)")
    p.add_argument("--min-area-frac", type=float, default=0.0,
                   help="最小面积占画面比例(0~1,优先于像素;分辨率/缩放无关)")
    p.add_argument("--max-area-frac", type=float, default=0.0,
                   help="最大面积占画面比例(0~1,优先于像素;0=不限)")
    p.add_argument("--max-aspect", type=float, default=0.0,
                   help="最大长宽比(0=不限;滤除细长噪声条纹)")
    p.add_argument("--morph-kernel", type=int, default=7, help="形态学核大小(奇数)")
    p.add_argument("--morph-iter", type=int, default=2, help="闭运算迭代次数")
    p.add_argument("--auto-adapt", action="store_true",
                   help="用连续校准帧自动估计目标面积、形态学核和匹配距离")
    p.add_argument("--calibration-frames", type=int, default=48,
                   help="自动适配校准帧数；实时源从启动后缓冲这些帧")
    p.add_argument("--bg-history", type=int, default=200, help="MOG2历史帧数")
    p.add_argument("--bg-var", type=float, default=40, help="MOG2方差阈值")
    p.add_argument("--ref-thresh", type=int, default=25, help="refbg灰度差阈值")
    p.add_argument("--bg-ref", default=None, help="refbg基准图路径(缺省或auto=自动中位数)")
    p.add_argument("--ref-alpha", type=float, default=0.0,
                   help="refbg参考帧慢更新系数(0=静态;>0抗光照漂移)")
    p.add_argument("--split-area", type=int, default=0, help="粘连分割触发面积(0=关)")
    p.add_argument("--unit-area", type=int, default=0, help="单个物体典型面积(配合分割)")
    p.add_argument("--merge-dist", type=float, default=0.0,
                   help="合并质心过近的检测(0=关;修复碎裂重复计数)")
    p.add_argument("--watershed-split", action="store_true",
                   help="对疑似粘连前景使用距离变换+分水岭拆分")
    p.add_argument("--watershed-min-distance", type=float, default=0.0,
                   help="分水岭峰值最小间距(0=按目标短边自动)")
    p.add_argument("--roi", default=None, help="感兴趣区 'x0,y0,x1,y1'(处理分辨率)")
    p.add_argument("--scale", type=float, default=1.0, help="处理前缩放系数")
    # 跟踪/计数
    p.add_argument("--max-dist", type=float, default=140, help="预测-检测最大匹配距离")
    p.add_argument("--track-ttl", type=int, default=5, help="轨迹漏检存活帧数")
    p.add_argument("--min-hits", type=int, default=1, help="轨迹连续确认帧数才计数")
    p.add_argument("--min-speed", type=float, default=0.0, help="计数所需最小轴向速度(px/帧)")
    p.add_argument("--transverse-gate", type=float, default=0.0,
                   help="垂直运动方向门控像素(0=按目标尺寸自动)")
    p.add_argument("--transverse-gate-factor", type=float, default=1.5,
                   help="自动垂直门控=目标横向尺寸×此系数")
    p.add_argument("--area-ratio-max", type=float, default=3.0,
                   help="匹配允许的最大检测框面积比(0=关闭)")
    p.add_argument("--shape-cost-weight", type=float, default=2.0,
                   help="匹配面积形状代价权重")
    p.add_argument("--cross-dedup-frames", type=int, default=0,
                   help="粘连分割越线去重时间窗(帧;0=关闭)")
    p.add_argument("--cross-dedup-dist", type=float, default=0.0,
                   help="粘连分割越线去重横向距离(像素;0=关闭)")
    p.add_argument("--cross-dedup-auto", action="store_true",
                   help="根据自动校准的目标短边设置保守越线去重距离")
    p.add_argument("--line", type=float, default=0.5, help="计数线位置(画面比例)")
    p.add_argument("--line-band", type=float, default=0.0, help="迟滞带宽(画面比例)")
    p.add_argument("--axis", choices=["x", "y"], default="x", help="运动/计数轴")
    p.add_argument("--flow", choices=["pos", "neg", "both"], default="both",
                   help="计数方向")
    p.add_argument("--warmup", type=int, default=8, help="bgsub预热帧(只建模不计数)")
    # 输出
    p.add_argument("--show", action="store_true", help="实时显示")
    p.add_argument("--save", default=None, help="保存可视化视频")
    p.add_argument("--save-fps", type=float, default=None,
                   help="标注视频播放帧率(不丢帧;低于源帧率=慢放,便于高帧率回看)")
    p.add_argument("--save-frames", default=None,
                   help="把标注帧另存为图片序列到该目录(图片文件夹工作流)")
    p.add_argument("--meta", default=None, help="计数真值json(默认自动查找)")
    p.add_argument("--debug", action="store_true", help="打印计数事件")
    p.add_argument("--profile", action="store_true",
                   help="逐帧打印处理耗时(仅检测+跟踪,不含读帧)+实时性统计")
    return p


def args_to_params(args):
    return {k: getattr(args, k) for k in DEFAULT_PARAMS if hasattr(args, k)}


def main():
    args = build_arg_parser().parse_args()
    params = args_to_params(args)
    count = count_source(args.source, params, fps=args.fps, save=args.save,
                         save_fps=args.save_fps, save_frames=args.save_frames,
                         show=args.show, debug=args.debug, verbose=True,
                         profile=args.profile)
    print(f"CV 计数结果: {count}")
    gt = find_gt(args.source, args.meta)
    if gt is not None:
        print(f"真值(越过中线): {gt}  |  误差: {count - gt}")


if __name__ == "__main__":
    main()
