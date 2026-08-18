import os
import re

import cv2


# Supported image formats readable by OpenCV's imread.
IMG_EXTS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib",
            ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm",
            ".jp2", ".ras", ".sr")


def natural_key(name):
    """Natural sort key: frame2 comes before frame10."""
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", name)]


class FrameSource:
    """Unified source for video files and ordered image directories."""

    def __init__(self, path, fps=30.0):
        self.path = path
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            self.files = sorted(
                (os.path.join(path, filename) for filename in os.listdir(path)
                 if filename.lower().endswith(IMG_EXTS)),
                key=lambda filename: natural_key(os.path.basename(filename)))
            if not self.files:
                raise RuntimeError(
                    f"directory contains no supported images ({'/'.join(IMG_EXTS)}): {path}")
            first = cv2.imread(self.files[0])
            if first is None:
                raise RuntimeError(f"unable to read image: {self.files[0]}")
            self.h, self.w = first.shape[:2]
            self.n = len(self.files)
            self.fps = fps
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"unable to open: {path}")
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def _read_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"unable to read image: {path}")
        actual_h, actual_w = image.shape[:2]
        if (actual_w, actual_h) != (self.w, self.h):
            raise RuntimeError(
                f"image size mismatch: {path}; expected {self.w}x{self.h}, "
                f"got {actual_w}x{actual_h}")
        return image

    def sample(self, i):
        if self.is_dir:
            return self._read_image(self.files[min(i, self.n - 1)])
        index = int(i)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            return frame
        if self.n > 0 and 0 <= index < self.n:
            raise RuntimeError(
                f"unable to read frame {index} from video: {self.path}")
        return None

    def frames(self):
        if self.is_dir:
            for path in self.files:
                yield self._read_image(path)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.n > 0:
                for index in range(self.n):
                    ret, frame = self.cap.read()
                    if not ret:
                        raise RuntimeError(
                            f"unable to read frame {index} from video: {self.path}")
                    yield frame
            else:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    yield frame


    def release(self):
        if not self.is_dir:
            self.cap.release()


def scaled(frame, scale, w, h):
    """Resize a frame on demand (return it unchanged when scale is 1)."""
    return frame if scale == 1.0 else cv2.resize(frame, (w, h))


def decode_all(source, scale=1.0, fps=30.0):
    """Decode a source into memory, optionally resizing each frame."""
    src = FrameSource(source, fps)
    w, h = int(src.w * scale), int(src.h * scale)
    frames = [scaled(frame, scale, w, h) for frame in src.frames()]
    src.release()
    return frames, w, h
