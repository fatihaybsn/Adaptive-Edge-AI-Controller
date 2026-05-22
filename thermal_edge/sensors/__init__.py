"""Sensor helpers for the Adaptive Edge MVP package."""

__all__ = [
    "KNOWN_GPU_LOAD_PATHS",
    "KNOWN_GPU_ZONE_TYPES",
    "discover_gpu_load_path",
    "discover_gpu_zone",
    "read_cpu_load",
    "read_gpu_load",
    "read_system_loads",
    "read_temperature",
    "read_gpu_temperature",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    from .thermal_zone import (
        KNOWN_GPU_ZONE_TYPES,
        discover_gpu_zone,
        read_temperature,
        read_gpu_temperature,
    )
    from .gpu_load import (
        KNOWN_GPU_LOAD_PATHS,
        discover_gpu_load_path,
        read_cpu_load,
        read_gpu_load,
        read_system_loads,
    )

    exports = {
        "KNOWN_GPU_LOAD_PATHS": KNOWN_GPU_LOAD_PATHS,
        "KNOWN_GPU_ZONE_TYPES": KNOWN_GPU_ZONE_TYPES,
        "discover_gpu_load_path": discover_gpu_load_path,
        "discover_gpu_zone": discover_gpu_zone,
        "read_cpu_load": read_cpu_load,
        "read_gpu_load": read_gpu_load,
        "read_system_loads": read_system_loads,
        "read_temperature": read_temperature,
        "read_gpu_temperature": read_gpu_temperature,
    }
    return exports[name]
