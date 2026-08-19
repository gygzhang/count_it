#!/usr/bin/env python3
from realtime.config import load_config
from realtime.hik_camera import HikCamera, load_mvs_sdk


def main():
    sdk = load_mvs_sdk()
    print(f"MVS Python SDK: {sdk.__file__}")
    camera = HikCamera(load_config().camera)
    try:
        devices = camera.enumerate_devices()
        if not devices:
            print("未发现相机")
        for index, device in enumerate(devices):
            print(f"[{index}] {device.label}  id={device.id}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
