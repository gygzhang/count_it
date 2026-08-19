#!/usr/bin/env python3
import argparse
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="海康高速相机实时计数 UI")
    parser.add_argument("--config", default="config/realtime.json")
    parser.add_argument("--video", default=None, help="不用相机，改用视频文件模拟")
    parser.add_argument("--no-loop", action="store_true", help="模拟视频到结尾后不循环")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit("缺少 PySide6，请执行: pip install -r requirements.txt") from exc

    from realtime.config import load_config
    from realtime.hik_camera import HikCamera
    from realtime.service import MeasurementService
    from realtime.ui import MainWindow
    from realtime.video_camera import VideoCamera

    config = load_config(args.config)
    if args.self_test:
        camera = HikCamera(config.camera, io_config=config.full_bin)
        try:
            devices = camera.enumerate_devices()
            if len(devices) != 1:
                return 2
            device = devices[0]
            if (
                device.model != config.camera.expected_model
                or device.serial != config.camera.expected_serial
                or device.transport != config.camera.expected_transport
            ):
                return 3
            camera.open(device.id)
            camera.set_digital_output(False)
            return 0
        except Exception:
            log_path = Path(sys.executable).with_suffix(".selftest.log")
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            return 4
        finally:
            camera.close()
    if args.video:
        factory = lambda: VideoCamera(args.video, loop=not args.no_loop, realtime=True)
        backend = "视频模拟"
    else:
        factory = lambda: HikCamera(config.camera, io_config=config.full_bin)
        backend = "海康 MVS"
    service = MeasurementService(factory, config)
    app = QApplication(sys.argv)
    window = MainWindow(service, config, backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
