"""GPU and CPU load reading helpers."""

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union


logger = logging.getLogger(__name__)

KNOWN_GPU_LOAD_PATHS: Tuple[Path, ...] = (
    Path("/sys/devices/gpu.0/load"),
    Path("/sys/devices/platform/gpu.0/load"),
)


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def discover_gpu_load_path(
    candidate_paths: Optional[Iterable[Union[str, Path]]] = None
) -> Optional[Path]:
    """Return the first available Jetson GPU load sysfs path."""

    paths = candidate_paths if candidate_paths is not None else KNOWN_GPU_LOAD_PATHS

    for candidate_path in paths:
        load_path = Path(candidate_path)
        try:
            if load_path.is_file():
                return load_path
        except (PermissionError, OSError) as exc:
            logger.debug("Unable to access GPU load path %s: %s", load_path, exc)

    return None


def read_gpu_load(path: Optional[Union[str, Path]] = None) -> float:
    """Read Jetson GPU load as a percentage."""

    load_path = Path(path) if path is not None else discover_gpu_load_path()
    if load_path is None:
        logger.debug("GPU load path was not found.")
        return 0.0

    try:
        raw_text = load_path.read_text(encoding="utf-8").strip()
        raw_value = float(raw_text)
    except FileNotFoundError:
        logger.warning("GPU load file was not found: %s", load_path)
        return 0.0
    except PermissionError:
        logger.warning("Permission denied while reading GPU load file: %s", load_path)
        return 0.0
    except ValueError:
        logger.warning("GPU load file contains an invalid value: %s", load_path)
        return 0.0
    except UnicodeDecodeError:
        logger.warning("GPU load file is not valid UTF-8: %s", load_path)
        return 0.0
    except OSError as exc:
        logger.warning("Unable to read GPU load file %s: %s", load_path, exc)
        return 0.0

    return _clamp_percent(raw_value / 10.0)


def read_cpu_load(interval: Optional[float] = None) -> float:
    """Read CPU load as a percentage."""

    try:
        import psutil
    except ImportError as exc:
        message = "psutil is required to read CPU load. Install it with: pip install psutil"
        logger.warning("%s", message)
        raise ImportError(message) from exc

    try:
        return _clamp_percent(float(psutil.cpu_percent(interval=interval)))
    except Exception as exc:
        logger.warning("Unable to read CPU load: %s", exc)
        return 0.0


def read_system_loads() -> Tuple[float, float]:
    """Read GPU and CPU load as percentages."""

    gpu_load = read_gpu_load()
    try:
        cpu_load = read_cpu_load()
    except ImportError:
        cpu_load = 0.0

    return gpu_load, cpu_load


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gpu_load_path = discover_gpu_load_path()
    gpu_load = read_gpu_load(gpu_load_path)
    try:
        cpu_load = read_cpu_load()
    except ImportError:
        cpu_load = 0.0

    print("GPU load path: {}".format(gpu_load_path))
    print("GPU load: {:.1f} %".format(gpu_load))
    print("CPU load: {:.1f} %".format(cpu_load))
