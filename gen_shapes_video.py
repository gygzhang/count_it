#!/usr/bin/env python3
"""
生成传送带/流水线风格的合成检测视频。

物体（不规则的圆 / 三角形 / 正方形 / 矩形，均带随机变形）从一侧匀速
移动到另一侧，模拟传送带上连续经过的物体。可用于目标检测 + 计数任务的
训练/测试数据生成，并可选导出 YOLO 格式真值标注与计数真值。

示例:
    # 生成 20 秒、画面同时约 6 个物体、速度 300 px/s 的视频
    python gen_shapes_video.py -o belt.mp4 --duration 20 --count 6 --speed 300

    # 同时导出 YOLO 标注 + 计数真值，并开启运动模糊 & 形状抖动
    python gen_shapes_video.py -o belt.mp4 --labels labels --motion-blur 9 --wobble 0.15
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

SHAPES = ["circle", "triangle", "square", "rectangle"]


def load_twemoji_assets(directory):
    """Load processed RGBA PNG assets from a directory."""
    if not directory:
        return []
    paths = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".png"):
            path = os.path.join(directory, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            if not np.any(img[:, :, 3] > 0):
                continue
            paths.append((os.path.splitext(name)[0], img))
    return paths


def base_unit_vertices(shape, rng):
    """返回以原点为中心、半径约 1 的基础顶点 (N,2)。"""
    if shape == "circle":
        n = 28
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    elif shape == "triangle":
        ang = np.deg2rad([90, 210, 330]) + rng.uniform(-0.2, 0.2)
        pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    elif shape == "square":
        pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) / math.sqrt(2)
    elif shape == "rectangle":
        ar = rng.uniform(1.4, 2.4)  # 长宽比
        pts = np.array([[-ar, -1], [ar, -1], [ar, 1], [-ar, 1]], float)
        pts = pts / np.linalg.norm(pts, axis=1).max()
    else:
        raise ValueError(shape)
    return pts


def deform(pts, amount, rng):
    """对每个顶点做随机径向 + 切向扰动，得到不规则形状。"""
    if amount <= 0:
        return pts
    radial = 1.0 + rng.uniform(-amount, amount, size=len(pts))
    out = pts * radial[:, None]
    out += rng.uniform(-amount * 0.5, amount * 0.5, size=out.shape)
    return out


def polygon_area(pts):
    """Return the absolute area of a polygon represented by Nx2 vertices."""
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1)))) * 0.5


def limit_size_range(min_size, max_size, area_ratio_limit=1.20):
    """Limit radius range so nominal max/min foreground area stays bounded."""
    if min_size <= 0 or max_size < min_size:
        raise ValueError("sizes must satisfy 0 < min_size <= max_size")
    if area_ratio_limit < 1:
        raise ValueError("area_ratio_limit must be >= 1")
    return min(max_size, min_size * math.sqrt(area_ratio_limit))


class Obj:
    """一个在传送带上移动的物体。"""

    def __init__(self, x, y, cfg, rng):
        self.asset = None
        if getattr(cfg, "twemoji_assets", None):
            # Optionally force a single named asset for controlled experiments.
            if getattr(cfg, "twemoji_object", None):
                matches = [a for a in cfg.twemoji_assets
                           if a[0] == cfg.twemoji_object]
                if not matches:
                    raise ValueError(
                        f"Twemoji object not found: {cfg.twemoji_object}")
                self.shape, self.asset = matches[0]
            else:
                self.shape, self.asset = cfg.twemoji_assets[rng.integers(len(cfg.twemoji_assets))]
        else:
            self.shape = SHAPES[rng.integers(len(SHAPES))]
        self.size = rng.uniform(cfg.min_size, cfg.effective_max_size)
        # Shape/scale/rotation are sampled once at creation. Objects are rigid
        # while moving; jitter parameters describe initial variation only.
        jitter = float(getattr(cfg, "scale_jitter", 0.0) or 0.0)
        aspect = max(0.1, 1.0 + rng.uniform(-jitter, jitter))
        # Reciprocal axes vary the initial shape without changing its area.
        self.aspect_x = aspect
        self.aspect_y = 1.0 / aspect
        if cfg.gray:
            contrast = max(1, int(cfg.gray_contrast))
            if cfg.gray_polarity == "dark":
                sign_gray = -1
            elif cfg.gray_polarity == "bright":
                sign_gray = 1
            else:
                sign_gray = -1 if rng.random() < 0.5 else 1
            v = int(np.clip(cfg.bg_gray + sign_gray * contrast, 0, 255))
            self.color = (v, v, v)
        else:
            # 彩色模式：HSV 高饱和度，保证物体明显区别于灰背景
            hsv = np.uint8([[[rng.integers(0, 180), rng.integers(150, 256),
                              rng.integers(140, 256)]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            self.color = tuple(int(c) for c in bgr)
        self.rot = rng.uniform(0, 2 * np.pi)
        if self.asset is not None:
            self.verts = None
        else:
            static_deform = min(0.5, max(0.0, cfg.deform + cfg.wobble))
            base = base_unit_vertices(self.shape, rng)
            self.verts = deform(base, static_deform, rng)
            self.verts[:, 0] *= self.aspect_x
            self.verts[:, 1] *= self.aspect_y
            # Normalize every generated shape to area=pi at unit size. Thus
            # only self.size controls object area, regardless of shape/deform.
            changed_area = polygon_area(self.verts)
            if changed_area > 0:
                self.verts *= math.sqrt(math.pi / changed_area)
        self.nominal_area = math.pi * self.size * self.size
        self.x = float(x)
        self.y = float(y)
        self.counted = False
        self.object_id = None
        self.overlap_group = None
        self.asset_alpha = None
        if self.asset is not None:
            self._prepare_asset()

    def _prepare_asset(self):
        """Apply the rigid scale/rotation once and cache a tight alpha mask."""
        rgba = self.asset
        h, w = rgba.shape[:2]
        alpha_mass = max(float(rgba[:, :, 3].sum()) / 255.0, 1.0)
        scale = math.sqrt(self.nominal_area / alpha_mass)
        nw = max(1, int(round(w * scale * self.aspect_x)))
        nh = max(1, int(round(h * scale * self.aspect_y)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        alpha = cv2.resize(rgba[:, :, 3], (nw, nh), interpolation=interp)
        pad = int(math.ceil(math.hypot(nw, nh))) + 2
        mat = cv2.getRotationMatrix2D(
            (nw / 2.0, nh / 2.0), self.rot * 180 / math.pi, 1.0)
        mat[0, 2] += (pad - nw) / 2.0
        mat[1, 2] += (pad - nh) / 2.0
        warped = cv2.warpAffine(
            alpha, mat, (pad, pad), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.where(warped > 0)
        if len(xs) == 0:
            raise ValueError("transformed asset has no foreground alpha")
        self.asset_alpha = warped[
            int(ys.min()):int(ys.max()) + 1,
            int(xs.min()):int(xs.max()) + 1,
        ]

    def half_extents(self):
        """Conservative axis-aligned half width/height of the rigid object."""
        if self.asset_alpha is not None:
            h, w = self.asset_alpha.shape
            return w / 2.0, h / 2.0
        pts = self.polygon(0.0, 0.0)
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(span[0] / 2.0), float(span[1] / 2.0)

    def dynamic_size(self, t, cfg):
        return self.size

    def dynamic_rot(self, t, cfg):
        return self.rot

    def polygon(self, t, wobble):
        v = self.verts
        c, s = math.cos(self.rot), math.sin(self.rot)
        rmat = np.array([[c, -s], [s, c]])
        v = v @ rmat.T
        pts = v * self.size + np.array([self.x, self.y])
        return pts

    def render_asset(self, frame, t, cfg):
        """Render RGBA asset and return alpha-derived bbox (x0,y0,x1,y1)."""
        alpha = self.asset_alpha
        # Place centered at object position.
        x0 = int(round(self.x - alpha.shape[1] / 2))
        y0 = int(round(self.y - alpha.shape[0] / 2))
        x1, y1 = x0 + alpha.shape[1], y0 + alpha.shape[0]
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(cfg.width, x1), min(cfg.height, y1)
        if fx0 < fx1 and fy0 < fy1:
            src_alpha = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
            dst = frame[fy0:fy1, fx0:fx1]
            a = src_alpha[:, :, None].astype(np.float32) / 255.0
            # Processed silhouettes carry zero RGB by design; tint them with
            # the object's generated color while compositing alpha only.
            tint = np.array(self.color, dtype=np.float32).reshape(1, 1, 3)
            dst[:] = (tint * a + dst.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        return x0, y0, x1, y1


def draw_belt(frame, offset, cfg):
    """画传送带背景。灰度模式为均匀灰底；彩色模式加移动接缝线。"""
    frame[:] = cfg.bg_color
    if cfg.gray:
        return  # 均匀灰底，靠亮度区分物体，不画会干扰分割的接缝线
    step = 80
    off = int(offset) % step
    if cfg.horizontal:
        for x in range(-step + off, cfg.width, step):
            cv2.line(frame, (x, 0), (x, cfg.height), (60, 60, 60), 2)
    else:
        for y in range(-step + off, cfg.height, step):
            cv2.line(frame, (0, y), (cfg.width, y), (60, 60, 60), 2)


def overlaps(y_or_x, existing, min_gap):
    return any(abs(y_or_x - e) < min_gap for e in existing)


def _object_mask(obj):
    """Return a tight uint8 mask and its top-left global origin."""
    if obj.asset_alpha is not None:
        return (obj.asset_alpha > 0).astype(np.uint8), (
            int(round(obj.x - obj.asset_alpha.shape[1] / 2)),
            int(round(obj.y - obj.asset_alpha.shape[0] / 2)),
        )
    pts = obj.polygon(0.0, 0.0)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    mask = np.zeros((max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)), np.uint8)
    cv2.fillPoly(mask, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
    return mask, (int(x0), int(y0))


def object_overlap_ratio(a, b):
    """Intersection area divided by the smaller object's foreground area."""
    ma, (ax, ay) = _object_mask(a)
    mb, (bx, by) = _object_mask(b)
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + ma.shape[1], bx + mb.shape[1])
    y1 = min(ay + ma.shape[0], by + mb.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    aa = ma[y0 - ay:y1 - ay, x0 - ax:x1 - ax]
    bb = mb[y0 - by:y1 - by, x0 - bx:x1 - bx]
    inter = float(np.count_nonzero(aa & bb))
    denom = max(1.0, float(min(np.count_nonzero(ma), np.count_nonzero(mb))))
    return inter / denom


def main():
    p = argparse.ArgumentParser(description="生成传送带合成检测视频")
    p.add_argument("-o", "--output", default="belt.mp4", help="输出视频路径")
    p.add_argument("--duration", type=float, default=15.0, help="时长(秒)")
    p.add_argument("--fps", type=int, default=30, help="帧率")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--count", type=int, default=6, help="画面中同时存在的物体数(约)")
    p.add_argument("--speed", type=float, default=500.0, help="移动速度(px/秒)")
    p.add_argument("--direction", choices=["l2r", "r2l", "t2b", "b2t"],
                   default="l2r", help="移动方向")
    p.add_argument("--gray", action="store_true",
                   help="灰度模式(物体靠亮度区分,均匀灰底,贴近工业相机)")
    p.add_argument("--bg-gray", type=int, default=120,
                   help="灰度模式背景值(0~255)")
    p.add_argument("--gray-contrast", type=int, default=70,
                   help="灰度目标与背景的绝对灰度差")
    p.add_argument("--gray-polarity", choices=["dark", "bright", "mixed"],
                   default="mixed", help="灰度目标为暗、亮或混合")
    p.add_argument("--noise", type=float, default=0.0, help="高斯传感器噪声标准差")
    p.add_argument("--min-size", type=float, default=28.0, help="物体半径下限(px)")
    p.add_argument("--max-size", type=float, default=48.0, help="物体半径上限(px)")
    p.add_argument("--deform", type=float, default=0.22, help="形状变形量(0~0.5)")
    p.add_argument("--wobble", type=float, default=0.0,
                   help="生成时附加一次性轮廓变形量(移动中保持不变)")
    p.add_argument("--scale-jitter", type=float, default=0.0,
                   help="物体缩放变形幅度(比例, 0关闭; 0.2=±20%%)")
    p.add_argument("--rotation-jitter", type=float, default=0.0,
                   help="初始旋转随机扰动强度(保留兼容；物体移动中不旋转)")
    p.add_argument("--motion-blur", type=int, default=0, help="运动模糊核大小(奇数,0关闭)")
    p.add_argument("--labels", default=None, help="YOLO 标注输出目录(不填则不导出)")
    p.add_argument("--outline", action="store_true", help="给物体画黑色描边")
    p.add_argument("--twemoji-assets", default=None,
                   help="processed Twemoji RGBA PNG directory; when set, use these assets instead of geometric shapes")
    p.add_argument("--twemoji-object", default=None,
                   help="仅使用指定 Twemoji 资源名称(不含 .png)，例如 apple")
    p.add_argument("--debug", action="store_true", help="打印每次跨线真值事件")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--empty-start", type=float, default=1.0,
                   help="开头无物体的时长(秒)，默认1秒")
    p.add_argument("--empty-end", type=float, default=1.0,
                   help="结尾无物体的时长(秒)，默认1秒")
    p.add_argument("--max-overlap", type=float, default=0.0,
                   help="允许两个物体前景重叠的最大比例(相对较小物体, 0~0.08)")
    p.add_argument("--overlap-rate", type=float, default=0.0,
                   help="尝试构造粘连的生成比例(0~1)，需配合 --max-overlap")
    cfg = p.parse_args()
    if cfg.min_size <= 0 or cfg.max_size < cfg.min_size:
        raise SystemExit("--min-size/--max-size must satisfy 0 < min-size <= max-size")
    # Foreground area scales approximately with radius². Restrict the radius
    # range so generated object areas differ by at most 20%.
    cfg.effective_max_size = limit_size_range(
        cfg.min_size, cfg.max_size, 1.20)
    cfg.area_ratio_limit = 1.20
    if not 0 <= cfg.max_overlap <= 0.08:
        raise SystemExit("--max-overlap must be in [0,0.08]")
    if not 0 <= cfg.overlap_rate <= 1:
        raise SystemExit("--overlap-rate must be in [0,1]")
    if cfg.overlap_rate > 0 and cfg.max_overlap <= 0:
        raise SystemExit("--overlap-rate requires --max-overlap > 0")
    if cfg.max_size > cfg.effective_max_size:
        print(
            f"[area-limit] max-size {cfg.max_size:g} -> "
            f"{cfg.effective_max_size:.3f} (area ratio <= 20%)")

    cfg.twemoji_assets = load_twemoji_assets(cfg.twemoji_assets)
    if cfg.twemoji_assets:
        print(f"已加载 Twemoji 资源: {len(cfg.twemoji_assets)}")
        if cfg.twemoji_object:
            names = {n for n, _ in cfg.twemoji_assets}
            if cfg.twemoji_object not in names:
                raise SystemExit(
                    f"未找到 Twemoji 对象 '{cfg.twemoji_object}'. "
                    f"可用示例: {', '.join(sorted(names)[:12])}")
    elif cfg.twemoji_object:
        raise SystemExit("--twemoji-object 需要同时提供 --twemoji-assets")

    cfg.horizontal = cfg.direction in ("l2r", "r2l")
    if not 0 <= cfg.bg_gray <= 255:
        raise SystemExit("--bg-gray must be in [0,255]")
    cfg.bg_color = ((cfg.bg_gray,) * 3) if cfg.gray else (110, 110, 110)
    rng = np.random.default_rng(cfg.seed)

    total_frames = int(round(cfg.duration * cfg.fps))
    speed_pf = cfg.speed / cfg.fps            # 每帧位移(px)
    span = cfg.width if cfg.horizontal else cfg.height
    spacing = max(span / max(cfg.count, 1), cfg.effective_max_size * 2)
    sign = 1 if cfg.direction in ("l2r", "t2b") else -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.output, fourcc, cfg.fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器: {cfg.output}")

    if cfg.labels:
        os.makedirs(cfg.labels, exist_ok=True)

    objs = []
    # Keep the beginning empty so a real detector can learn the background.
    # The first object is spawned only after this warmup interval.
    dist_since_spawn = 0.0
    warmup_frames = max(0, int(round(cfg.empty_start * cfg.fps)))
    end_frames = max(0, int(round(cfg.empty_end * cfg.fps)))
    margin = cfg.effective_max_size * 1.5
    # Stop spawning early enough for the last object to leave the frame.
    travel_frames = int(math.ceil((span + 2 * margin) / max(speed_pf, 1e-6)))
    spawn_stop_frame = max(warmup_frames, total_frames - end_frames - travel_frames)
    total_spawned = 0
    nominal_areas = []
    overlap_events = 0
    max_observed_overlap = 0.0
    crossed = 0  # 越过中线的数量(计数真值)
    next_object_id = 0
    next_overlap_group = 0
    cross_at = span / 2.0

    def spawn():
        nonlocal total_spawned, overlap_events, max_observed_overlap
        nonlocal next_object_id, next_overlap_group
        # 主轴起点位置
        if cfg.horizontal:
            main = -margin if sign > 0 else cfg.width + margin
        else:
            main = -margin if sign > 0 else cfg.height + margin
        # Sample a rigid object first, then place it only if its transformed
        # axis-aligned bounds do not intersect any existing object.
        candidate = Obj(0, 0, cfg, rng)
        chw, chh = candidate.half_extents()
        cross_half = chh if cfg.horizontal else chw
        cross_span = cfg.height if cfg.horizontal else cfg.width
        cross_lo, cross_hi = cross_half + 2, cross_span - cross_half - 2
        if cross_lo >= cross_hi:
            return False
        placed = False
        for _ in range(64):
            pos = rng.uniform(cross_lo, cross_hi)
            x, y = (main, pos) if cfg.horizontal else (pos, main)
            candidate.x, candidate.y = float(x), float(y)
            collision = False
            for other in objs:
                ohw, ohh = other.half_extents()
                bbox_hit = (abs(candidate.x - other.x) < chw + ohw + 2 and
                            abs(candidate.y - other.y) < chh + ohh + 2)
                ratio = object_overlap_ratio(candidate, other) if bbox_hit else 0.0
                allowed = cfg.max_overlap if cfg.max_overlap > 0 else 0.0
                if bbox_hit and ratio > allowed:
                    collision = True
                    break
                if ratio > 0:
                    max_observed_overlap = max(max_observed_overlap, ratio)
            if not collision:
                placed = True
                break
        if not placed:
            return False
        candidate.object_id = next_object_id
        next_object_id += 1
        objs.append(candidate)
        nominal_areas.append(candidate.nominal_area)
        total_spawned += 1

        # Optionally create a companion touching this newly spawned object.
        # Both are born at the entry together and then move as rigid bodies.
        if cfg.max_overlap > 0 and rng.random() < cfg.overlap_rate:
            companion = Obj(candidate.x, candidate.y, cfg, rng)
            cwh, chh = companion.half_extents()
            base_cross_half = chh if cfg.horizontal else cwh
            target_cross_half = chh + candidate.half_extents()[1] if cfg.horizontal else cwh + candidate.half_extents()[0]
            for _ in range(128):
                # A small penetration around the sum of half extents normally
                # creates a connected pair without substantial occlusion.
                penetration = rng.uniform(0.25, max(0.5, base_cross_half * 0.12))
                offset = target_cross_half - penetration
                if rng.random() < 0.5:
                    offset = -offset
                if cfg.horizontal:
                    companion.x = candidate.x
                    companion.y = candidate.y + offset
                    in_bounds = cross_lo <= companion.y <= cross_hi
                else:
                    companion.x = candidate.x + offset
                    companion.y = candidate.y
                    in_bounds = cross_lo <= companion.x <= cross_hi
                if not in_bounds:
                    continue
                ratio = object_overlap_ratio(candidate, companion)
                if not (0 < ratio <= cfg.max_overlap):
                    continue
                if any(object_overlap_ratio(companion, other) > cfg.max_overlap
                       for other in objs[:-1]):
                    continue
                companion.object_id = next_object_id
                next_object_id += 1
                candidate.overlap_group = next_overlap_group
                companion.overlap_group = next_overlap_group
                next_overlap_group += 1
                objs.append(companion)
                nominal_areas.append(companion.nominal_area)
                total_spawned += 1
                overlap_events += 1
                max_observed_overlap = max(max_observed_overlap, ratio)
                break
        return True

    # 运动模糊核只依赖固定参数，循环外构建一次
    blur_kernel = None
    if cfg.motion_blur and cfg.motion_blur >= 3:
        bk = cfg.motion_blur | 1
        blur_kernel = np.zeros((bk, bk), np.float32)
        if cfg.horizontal:
            blur_kernel[bk // 2, :] = 1.0 / bk
        else:
            blur_kernel[:, bk // 2] = 1.0 / bk

    for f in range(total_frames):
        t = f / cfg.fps
        frame = np.empty((cfg.height, cfg.width, 3), np.uint8)
        draw_belt(frame, sign * speed_pf * f, cfg)

        # 移动
        for o in objs:
            if cfg.horizontal:
                o.x += sign * speed_pf
            else:
                o.y += sign * speed_pf

        # 生成新物体维持画面数量, after the empty warmup.
        if warmup_frames <= f < spawn_stop_frame:
            dist_since_spawn += speed_pf
            while dist_since_spawn >= spacing:
                # If the entry region is full, retain the spawn distance and
                # retry on the next frame rather than forcing an overlap.
                if not spawn():
                    break
                dist_since_spawn -= spacing

        # 绘制 + 收集标注
        lines = []
        frame_objects = []
        for o in objs:
            if o.asset is not None:
                bbox = o.render_asset(frame, t, cfg)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
            else:
                fpts = o.polygon(t, cfg.wobble)   # 只算一次，画框与bbox共用
                pts = fpts.astype(np.int32)
                cv2.fillPoly(frame, [pts], o.color)
                if cfg.outline:
                    cv2.polylines(frame, [pts], True, (20, 20, 20), 2)
                x0, y0 = fpts.min(axis=0)
                x1, y1 = fpts.max(axis=0)
            cx = (x0 + x1) / 2 / cfg.width
            cy = (y0 + y1) / 2 / cfg.height
            bw = (x1 - x0) / cfg.width
            bh = (y1 - y0) / cfg.height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            frame_objects.append({
                "object_id": o.object_id,
                "overlap_group": o.overlap_group,
                "asset": o.shape,
                "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "center": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
                "counted": bool(o.counted),
            })

            # 计数真值：物体中心越过中线
            main_pos = o.x if cfg.horizontal else o.y
            if not o.counted and ((sign > 0 and main_pos >= cross_at) or
                                  (sign < 0 and main_pos <= cross_at)):
                o.counted = True
                crossed += 1
                if cfg.debug:
                    print(f"[GT cross {crossed}] frame={f} shape={o.shape} "
                          f"main_pos={main_pos:.1f}")

        # 传感器噪声（模拟真实相机）
        if cfg.noise > 0:
            n = rng.normal(0, cfg.noise, frame.shape)
            frame = np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)

        # 运动模糊(沿运动方向)
        if blur_kernel is not None:
            frame = cv2.filter2D(frame, -1, blur_kernel)

        writer.write(frame)

        if cfg.labels:
            with open(os.path.join(cfg.labels, f"frame_{f:06d}.txt"), "w") as fh:
                fh.write("\n".join(lines))
            with open(os.path.join(cfg.labels, f"frame_{f:06d}.json"), "w") as fh:
                json.dump({"frame": f, "objects": frame_objects}, fh)

        # 移除完全离开画面的物体
        kept = []
        for o in objs:
            m = o.x if cfg.horizontal else o.y
            if -margin * 2 <= m <= (span + margin * 2):
                kept.append(o)
        objs = kept

    writer.release()

    meta = {
        "output": cfg.output,
        "frames": total_frames,
        "fps": cfg.fps,
        "resolution": [cfg.width, cfg.height],
        "duration_s": cfg.duration,
        "speed_px_s": cfg.speed,
        "direction": cfg.direction,
        "target_on_screen": cfg.count,
        "total_spawned": total_spawned,
        "crossed_center_line": crossed,
        "twemoji_object": cfg.twemoji_object,
        "scale_jitter": cfg.scale_jitter,
        "rotation_jitter": cfg.rotation_jitter,
        "wobble": cfg.wobble,
        "empty_start_s": cfg.empty_start,
        "empty_end_s": cfg.empty_end,
        "effective_min_size": cfg.min_size,
        "effective_max_size": cfg.effective_max_size,
        "area_ratio_limit": cfg.area_ratio_limit,
        "bg_gray": cfg.bg_gray,
        "gray_contrast": cfg.gray_contrast,
        "gray_polarity": cfg.gray_polarity,
        "nominal_area_min_px2": min(nominal_areas) if nominal_areas else 0,
        "nominal_area_max_px2": max(nominal_areas) if nominal_areas else 0,
        "nominal_area_ratio": (
            max(nominal_areas) / min(nominal_areas)
            if nominal_areas and min(nominal_areas) > 0 else 1.0),
        "max_overlap_ratio": cfg.max_overlap,
        "overlap_rate": cfg.overlap_rate,
        "overlap_events": overlap_events,
        "max_observed_overlap_ratio": max_observed_overlap,
    }
    meta_path = os.path.splitext(cfg.output)[0] + "_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"视频已生成: {cfg.output}")
    print(f"计数真值(越过中线): {crossed}  |  总生成: {total_spawned}")
    print(f"元数据: {meta_path}")
    if cfg.labels:
        print(f"YOLO 标注: {cfg.labels}/frame_*.txt")


if __name__ == "__main__":
    main()
