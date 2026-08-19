"""HIKROBOT MVS SDK adapter.

The vendor's ``MvImport`` Python wrapper is loaded at runtime.  It remains a
deployment dependency so vendor SDK binaries are not copied into this project.
"""
import ctypes
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from .camera_base import CameraSource
from .types import CameraDevice, FramePacket


class HikSdkUnavailable(RuntimeError):
    pass


_DLL_DIRECTORY_HANDLES = []


def _prepare_mvs_dll_search():
    """Expose the installed 64-bit MVS runtime to frozen applications."""
    runtime = os.getenv("MVCAM_COMMON_RUNENV")
    candidates = [
        runtime,
        str(Path(runtime) / "Win64_x64") if runtime else None,
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
        r"D:\MVS\Runtime\Win64_x64",
    ]
    for value in candidates:
        if not value:
            continue
        directory = Path(value)
        if not (directory / "MvCameraControl.dll").exists():
            continue
        directory_text = str(directory)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
            except OSError:
                pass
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text not in path_parts:
            os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
        return directory_text
    return None


def _sdk_search_paths(explicit=None):
    runtime_or_development = os.getenv("MVCAM_COMMON_RUNENV")
    values = [
        explicit,
        os.getenv("MVS_PYTHON_SDK"),
        runtime_or_development,
        str(Path(runtime_or_development) / "Samples" / "Python")
        if runtime_or_development else None,
    ]
    values += [
        r"C:\Program Files (x86)\MVS\Development\Samples\Python",
        r"C:\Program Files\MVS\Development\Samples\Python",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\Samples\Python",
        str(Path(__file__).resolve().parents[1] / "vendor"),
    ]
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.name == "MvImport":
            yield path.parent
            yield path
        else:
            yield path
            yield path / "MvImport"


def load_mvs_sdk(explicit=None):
    _prepare_mvs_dll_search()
    errors = []
    for path in _sdk_search_paths(explicit):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for module_name in ("MvImport.MvCameraControl_class", "MvCameraControl_class"):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise HikSdkUnavailable(
        "未找到海康 MVS Python SDK。请安装 MVS 的 Development/Python Samples，"
        "或把包含 MvImport 的目录写入环境变量 MVS_PYTHON_SDK。\n" +
        "\n".join(errors)
    )


def _c_text(value):
    try:
        raw = bytes(value)
    except TypeError:
        raw = bytes(bytearray(value))
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


class HikCamera(CameraSource):
    def __init__(self, config, sdk_path=None, io_config=None):
        self.config = config
        self.io_config = io_config
        self.sdk_path = sdk_path
        self.sdk = None
        self.camera = None
        self.running = False
        self.warnings = []

    def _load(self):
        if self.sdk is None:
            self.sdk = load_mvs_sdk(self.sdk_path)
        return self.sdk

    @staticmethod
    def _ret_hex(ret):
        return f"0x{ctypes.c_uint32(ret).value:08X}"

    def _check(self, ret, action):
        if ret != 0:
            raise RuntimeError(f"{action}失败: {self._ret_hex(ret)}")

    def _enum_raw(self):
        sdk = self._load()
        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        mask = sdk.MV_USB_DEVICE | sdk.MV_GIGE_DEVICE
        self._check(sdk.MvCamera.MV_CC_EnumDevices(mask, devices), "枚举相机")
        result = []
        for index in range(int(devices.nDeviceNum)):
            info = ctypes.cast(
                devices.pDeviceInfo[index], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)
            ).contents
            if info.nTLayerType == sdk.MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                model = _c_text(usb.chModelName)
                serial = _c_text(usb.chSerialNumber)
                transport = "USB3"
            else:
                gige = info.SpecialInfo.stGigEInfo
                model = _c_text(gige.chModelName)
                serial = _c_text(gige.chSerialNumber)
                transport = "GigE"
            device_id = f"{serial}|{index}"
            device = CameraDevice(device_id, model, serial, transport)
            expected_model = str(getattr(self.config, "expected_model", "") or "")
            expected_serial = str(getattr(self.config, "expected_serial", "") or "")
            expected_transport = str(
                getattr(self.config, "expected_transport", "") or ""
            )
            if expected_model and device.model != expected_model:
                continue
            if expected_serial and device.serial != expected_serial:
                continue
            if expected_transport and device.transport != expected_transport:
                continue
            result.append((device, info))
        return result

    def enumerate_devices(self):
        return [item[0] for item in self._enum_raw()]

    def open(self, device_id):
        sdk = self._load()
        matches = self._enum_raw()
        selected = next((entry for entry in matches if entry[0].id == device_id), None)
        if selected is None:
            serial = device_id.split("|", 1)[0]
            selected = next((entry for entry in matches if entry[0].serial == serial), None)
        if selected is None:
            raise RuntimeError(f"相机已断开或设备编号无效: {device_id}")

        self.camera = sdk.MvCamera()
        self._check(self.camera.MV_CC_CreateHandle(selected[1]), "创建相机句柄")
        try:
            access = getattr(sdk, "MV_ACCESS_Exclusive", 1)
            self._check(self.camera.MV_CC_OpenDevice(access, 0), "打开相机")
            self._configure()
        except Exception:
            self.close()
            raise

    def _optional(self, method, args, label):
        fn = getattr(self.camera, method, None)
        if fn is None:
            self.warnings.append(f"当前 SDK 不支持 {label}")
            return
        ret = fn(*args)
        if ret != 0:
            self.warnings.append(f"{label}设置失败 {self._ret_hex(ret)}")

    def _enum_symbols(self, key):
        """Return enum symbolic names exposed by this camera's GenICam XML."""
        value = self.sdk.MVCC_ENUMVALUE_EX()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        ret = self.camera.MV_CC_GetEnumValueEx(key, value)
        if ret != 0:
            return []
        result = []
        for index in range(int(value.nSupportedNum)):
            entry = self.sdk.MVCC_ENUMENTRY()
            ctypes.memset(ctypes.byref(entry), 0, ctypes.sizeof(entry))
            entry.nValue = int(value.nSupportValue[index])
            if self.camera.MV_CC_GetEnumEntrySymbolic(key, entry) == 0:
                result.append(
                    bytes(entry.chSymbolic).split(b"\0", 1)[0].decode(
                        "ascii", errors="replace"
                    )
                )
        return result

    def _set_output_polarity(self, active_high):
        if active_high:
            self._optional(
                "MV_CC_SetBoolValue", ("LineInverter", False), "设置 IO 高电平有效"
            )
        else:
            self._check(
                self.camera.MV_CC_SetBoolValue("LineInverter", True),
                "设置 IO 低电平有效",
            )

    def _configure(self):
        sdk = self.sdk
        self.warnings.clear()
        self._check(
            self.camera.MV_CC_SetEnumValue(
                "TriggerMode", getattr(sdk, "MV_TRIGGER_MODE_OFF", 0)
            ),
            "关闭触发模式",
        )
        # Mono8 avoids conversion and is the lightest format for this mono camera.
        mono8 = getattr(sdk, "PixelType_Gvsp_Mono8", 0x01080001)
        self._check(self.camera.MV_CC_SetEnumValue("PixelFormat", mono8), "设置 Mono8")
        self._optional(
            "MV_CC_SetBoolValue", ("AcquisitionFrameRateEnable", False),
            "关闭帧率限速",
        )
        self._optional(
            "MV_CC_SetImageNodeNum", (int(self.config.buffer_nodes),),
            f"SDK缓存节点({self.config.buffer_nodes})",
        )
        strategy = getattr(sdk, "MV_GrabStrategy_OneByOne", 0)
        self._optional("MV_CC_SetGrabStrategy", (strategy,), "逐帧取流策略")

        if self.config.exposure_us > 0:
            self._optional("MV_CC_SetEnumValue", ("ExposureAuto", 0), "关闭自动曝光")
            self._optional(
                "MV_CC_SetFloatValue", ("ExposureTime", float(self.config.exposure_us)),
                "曝光时间",
            )
        if self.config.gain_db >= 0:
            self._optional("MV_CC_SetEnumValue", ("GainAuto", 0), "关闭自动增益")
            self._optional(
                "MV_CC_SetFloatValue", ("Gain", float(self.config.gain_db)), "增益"
            )

    def start(self):
        if self.camera is None:
            raise RuntimeError("相机尚未打开")
        self._check(self.camera.MV_CC_StartGrabbing(), "开始取流")
        self.running = True

    def set_digital_output(self, active):
        """Drive a persistent UserOutput level through the configured camera line."""
        if self.camera is None:
            raise RuntimeError("相机尚未打开，无法设置 IO 输出")
        if self.io_config is None:
            raise RuntimeError("未提供相机 IO 配置")
        io = self.io_config
        self._check(
            self.camera.MV_CC_SetEnumValueByString(
                "LineSelector", io.line_selector
            ),
            f"选择输出线路 {io.line_selector}",
        )
        sources = self._enum_symbols("LineSource")
        if io.user_output_selector in sources:
            # Cameras with a software-controlled UserOutput node.
            self._optional(
                "MV_CC_SetEnumValueByString", ("LineMode", "Output"),
                "设置线路为输出",
            )
            self._check(
                self.camera.MV_CC_SetEnumValueByString(
                    "LineSource", io.user_output_selector
                ),
                f"设置输出源 {io.user_output_selector}",
            )
            self._set_output_polarity(io.active_high)
            self._check(
                self.camera.MV_CC_SetEnumValueByString(
                    "UserOutputSelector", io.user_output_selector
                ),
                f"选择用户输出 {io.user_output_selector}",
            )
            self._check(
                self.camera.MV_CC_SetBoolValue("UserOutputValue", bool(active)),
                "设置用户输出电平",
            )
            return True

        # This camera exposes Line1 only as an opto-isolated Strobe output.
        # A source event would only produce a short pulse, so use the line
        # inverter with Strobe disabled to hold the open-collector transistor
        # steadily on/off for wiring tests and full-bin control.
        strobe_source = "ExposureStartActive"
        if strobe_source not in sources:
            available = ", ".join(sources) or "无可读取项"
            raise RuntimeError(
                f"Line1 不支持可控的 Strobe 输出；可用源: {available}"
            )
        self._optional(
            "MV_CC_SetEnumValueByString", ("LineMode", "Strobe"),
            "设置 Line1 为 Strobe",
        )
        self._check(
            self.camera.MV_CC_SetEnumValueByString("LineSource", strobe_source),
            f"设置输出源 {strobe_source}",
        )
        self._check(
            self.camera.MV_CC_SetBoolValue("StrobeEnable", False),
            "关闭 Line1 短脉冲输出",
        )
        # Non-inverted idle is open/high. The NPN input wiring used here is
        # active-low, so inversion holds the opto transistor in the sink state.
        inverter = bool(active) if not io.active_high else not bool(active)
        self._check(
            self.camera.MV_CC_SetBoolValue("LineInverter", inverter),
            "切换 Line1 保持输出",
        )
        return True

    def read(self, timeout_ms=100):
        if not self.running or self.camera is None:
            return None
        sdk = self.sdk
        output = sdk.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(output), 0, ctypes.sizeof(output))
        ret = self.camera.MV_CC_GetImageBuffer(output, int(timeout_ms))
        if ret != 0:
            no_data = ctypes.c_uint32(getattr(sdk, "MV_E_NODATA", 0x80000007)).value
            if ctypes.c_uint32(ret).value == no_data:
                return None
            raise RuntimeError(f"取图失败: {self._ret_hex(ret)}")
        try:
            info = output.stFrameInfo
            mono8 = getattr(sdk, "PixelType_Gvsp_Mono8", 0x01080001)
            if int(info.enPixelType) != int(mono8):
                raise RuntimeError(
                    f"相机输出不是 Mono8: 0x{int(info.enPixelType):08X}"
                )
            length = int(info.nFrameLen)
            # One copy from the SDK-owned node into application memory.  A
            # string_at()+numpy.copy() sequence would copy every frame twice.
            raw_view = np.ctypeslib.as_array(output.pBufAddr, shape=(length,))
            image = raw_view.reshape(int(info.nHeight), int(info.nWidth)).copy()
            return FramePacket(
                image=image,
                frame_no=int(info.nFrameNum),
                captured_ns=time.perf_counter_ns(),
                camera_timestamp=(int(getattr(info, "nDevTimeStampHigh", 0)) << 32)
                | int(getattr(info, "nDevTimeStampLow", 0)),
            )
        finally:
            self.camera.MV_CC_FreeImageBuffer(output)

    def stop(self):
        if self.camera is not None and self.running:
            self.camera.MV_CC_StopGrabbing()
        self.running = False

    def close(self):
        try:
            self.stop()
        finally:
            if self.camera is not None:
                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
                self.camera = None
