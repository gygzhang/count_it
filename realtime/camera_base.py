from abc import ABC, abstractmethod
from typing import List, Optional

from .types import CameraDevice, FramePacket


class CameraSource(ABC):
    @abstractmethod
    def enumerate_devices(self) -> List[CameraDevice]:
        raise NotImplementedError

    @abstractmethod
    def open(self, device_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, timeout_ms: int) -> Optional[FramePacket]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def set_digital_output(self, active: bool) -> bool:
        """Set a persistent camera output; return False when unsupported."""
        return False
