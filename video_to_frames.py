#!/usr/bin/env python3
"""
把视频抽帧成图片，保存到文件夹（供逐帧算法处理）。

示例:
    python video_to_frames.py belt.mp4 frames_belt
    python video_to_frames.py belt.mp4 frames_belt --ext png --step 2
"""
import argparse
import os
import shutil

import cv2


def timestamp_name(frame_index, fps):
    """Return a Windows-safe HH-MM-SS.mmm timestamp for a constant-FPS video."""
    milliseconds = int(round(frame_index * 1000.0 / fps)) if fps > 0 else 0
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}.{milliseconds:03d}"


def main():
    p = argparse.ArgumentParser(description="视频抽帧成图片文件夹")
    p.add_argument("video", help="输入视频")
    p.add_argument("out_dir", help="输出图片文件夹")
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="图片格式")
    p.add_argument("--step", type=int, default=1, help="每隔 N 帧抽一张")
    p.add_argument("--start", type=int, default=0, help="起始帧")
    p.add_argument("--end", type=int, default=-1, help="结束帧(-1到末尾)")
    p.add_argument("--quality", type=int, default=95, help="jpg 质量(1-100)")
    p.add_argument("--no-timestamp", dest="timestamp", action="store_false",
                   help="Do not add video time to output filenames")
    p.set_defaults(timestamp=True)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开: {args.video}")
    os.makedirs(args.out_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    params = []
    if args.ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    idx = -1
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx < args.start:
            continue
        if args.end >= 0 and idx > args.end:
            break
        if (idx - args.start) % args.step != 0:
            continue
        suffix = f"_t{timestamp_name(idx, fps)}" if args.timestamp else ""
        path = os.path.join(args.out_dir, f"frame_{idx:06d}{suffix}.{args.ext}")
        cv2.imwrite(path, frame, params)
        saved += 1
    cap.release()

    # 若存在计数真值，拷贝到图片文件夹，方便后续核对
    meta = os.path.splitext(args.video)[0] + "_meta.json"
    if os.path.exists(meta):
        shutil.copy(meta, os.path.join(args.out_dir, os.path.basename(meta)))

    print(f"已抽帧 {saved} 张 -> {args.out_dir}/frame_*.{args.ext}")


if __name__ == "__main__":
    main()
