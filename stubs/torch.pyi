"""Type stubs for PyTorch."""

from types import ModuleType
from typing import Any, Optional, Tuple, Union

class DeviceType:
    index: int
    type: str

class CudaDevice:
    def get_device_properties(self, device: int) -> Any: ...
    def is_available(self) -> bool: ...
    def get_device_name(self, device: int) -> str: ...
    def memory_allocated(self, device: int) -> int: ...
    def memory_reserved(self, device: int) -> int: ...

class Tensor:
    def to(self, device: Union[str, DeviceType]) -> "Tensor": ...
    def cuda(self) -> "Tensor": ...
    def cpu(self) -> "Tensor": ...

class Backends:
    cudnn: Any

class dtype:
    """PyTorch data type."""

    pass

# Common dtypes
float16: dtype
float32: dtype
int8: dtype
int16: dtype
int32: dtype
int64: dtype

cuda: CudaDevice
backends: Backends

def load(f: Any, map_location: Any = None) -> Any: ...
def no_grad() -> Any: ...
def inference_mode() -> Any: ...
