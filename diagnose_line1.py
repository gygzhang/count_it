#!/usr/bin/env python3
"""List the target camera's available Line1 output enum symbols."""
import ctypes

from realtime.config import load_config
from realtime.hik_camera import HikCamera


def enum_symbols(camera, key):
    value = camera.sdk.MVCC_ENUMVALUE_EX()
    ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
    ret = camera.camera.MV_CC_GetEnumValueEx(key, value)
    if ret != 0:
        return ret, []
    symbols = []
    for index in range(int(value.nSupportedNum)):
        entry = camera.sdk.MVCC_ENUMENTRY()
        ctypes.memset(ctypes.byref(entry), 0, ctypes.sizeof(entry))
        entry.nValue = int(value.nSupportValue[index])
        entry_ret = camera.camera.MV_CC_GetEnumEntrySymbolic(key, entry)
        if entry_ret == 0:
            name = bytes(entry.chSymbolic).split(b"\0", 1)[0].decode("ascii", "replace")
            symbols.append((int(entry.nValue), name))
    return ret, symbols


def main():
    config = load_config()
    camera = HikCamera(config.camera, io_config=config.full_bin)
    try:
        devices = camera.enumerate_devices()
        if not devices:
            raise RuntimeError("未发现相机")
        print("camera:", devices[0].label)
        camera.open(devices[0].id)
        ret = camera.camera.MV_CC_SetEnumValueByString("LineSelector", "Line1")
        print("select Line1:", camera._ret_hex(ret))
        for key in ("LineMode", "LineSource", "UserOutputSelector"):
            ret, values = enum_symbols(camera, key)
            print(f"{key}: {camera._ret_hex(ret)} {values}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
