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


class Obj:
    """一个在传送带上移动的物体。"""

    def __init__(self, x, y, cfg, rng):
        self.shape = SHAPES[rng.integers(len(SHAPES))]
        self.size = rng.uniform(cfg.min_size, cfg.max_size)
        if cfg.gray:
            # 灰度模式：物体取明显偏暗或偏亮的灰阶，保证与灰背景有对比
            if rng.random() < 0.5:
                v = int(rng.integers(15, 70))    # 暗物体
            else:
                v = int(rng.integers(185, 245))  # 亮物体
            self.color = (v, v, v)
        else:
            # 彩色模式：HSV 高饱和度，保证物体明显区别于灰背景
            hsv = np.uint8([[[rng.integers(0, 180), rng.integers(150, 256),
                              rng.integers(140, 256)]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            self.color = tuple(int(c) for c in bgr)
        self.rot = rng.uniform(0, 2 * np.pi)
        self.verts = deform(base_unit_vertices(self.shape, rng), cfg.deform, rng)
        self.phase = rng.uniform(0, 2 * np.pi, size=len(self.verts))
        self.x = float(x)
        self.y = float(y)
        self.counted = False

    def polygon(self, t, wobble):
        v = self.verts
        if wobble > 0:
            v = v * (1.0 + wobble * np.sin(2 * np.pi * t + self.phase))[:, None]
        c, s = math.cos(self.rot), math.sin(self.rot)
        rmat = np.array([[c, -s], [s, c]])
        v = v @ rmat.T
        pts = v * self.size + np.array([self.x, self.y])
        return pts


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
    p.add_argument("--noise", type=float, default=0.0, help="高斯传感器噪声标准差")
    p.add_argument("--min-size", type=float, default=28.0, help="物体半径下限(px)")
    p.add_argument("--max-size", type=float, default=48.0, help="物体半径上限(px)")
    p.add_argument("--deform", type=float, default=0.22, help="形状变形量(0~0.5)")
    p.add_argument("--wobble", type=float, default=0.0, help="逐帧抖动幅度(0关闭)")
    p.add_argument("--motion-blur", type=int, default=0, help="运动模糊核大小(奇数,0关闭)")
    p.add_argument("--labels", default=None, help="YOLO 标注输出目录(不填则不导出)")
    p.add_argument("--outline", action="store_true", help="给物体画黑色描边")
    p.add_argument("--debug", action="store_true", help="打印每次跨线真值事件")
    p.add_argument("--seed", type=int, default=0)
    cfg = p.parse_args()

    cfg.horizontal = cfg.direction in ("l2r", "r2l")
    cfg.bg_color = (120, 120, 120) if cfg.gray else (110, 110, 110)
    rng = np.random.default_rng(cfg.seed)

    total_frames = int(round(cfg.duration * cfg.fps))
    speed_pf = cfg.speed / cfg.fps            # 每帧位移(px)
    span = cfg.width if cfg.horizontal else cfg.height
    spacing = max(span / max(cfg.count, 1), cfg.max_size * 2)
    sign = 1 if cfg.direction in ("l2r", "t2b") else -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.output, fourcc, cfg.fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器: {cfg.output}")

    if cfg.labels:
        os.makedirs(cfg.labels, exist_ok=True)

    objs = []
    dist_since_spawn = spacing  # 让第一帧就开始生成
    margin = cfg.max_size * 1.5
    total_spawned = 0
    crossed = 0  # 越过中线的数量(计数真值)
    cross_at = span / 2.0

    def spawn():
        nonlocal total_spawned
        # 主轴起点位置
        if cfg.horizontal:
            main = -margin if sign > 0 else cfg.width + margin
        else:
            main = -margin if sign > 0 else cfg.height + margin
        # 副轴(垂直于运动方向)位置，尽量不与刚生成的物体重叠
        cross_lo, cross_hi = margin, (cfg.height if cfg.horizontal else cfg.width) - margin
        recent = [(o.y if cfg.horizontal else o.x) for o in objs
                  if abs((o.x if cfg.horizontal else o.y) - main) < spacing]
        pos = None
        for _ in range(12):
            cand = rng.uniform(cross_lo, cross_hi)
            if not overlaps(cand, recent, cfg.max_size * 2):
                pos = cand
                break
        if pos is None:
            pos = rng.uniform(cross_lo, cross_hi)
        x, y = (main, pos) if cfg.horizontal else (pos, main)
        objs.append(Obj(x, y, cfg, rng))
        total_spawned += 1

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

        # 生成新物体维持画面数量
        dist_since_spawn += speed_pf
        while dist_since_spawn >= spacing:
            spawn()
            dist_since_spawn -= spacing

        # 绘制 + 收集标注
        lines = []
        for o in objs:
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
