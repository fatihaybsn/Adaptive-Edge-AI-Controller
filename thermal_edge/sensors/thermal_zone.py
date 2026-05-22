"""GPU thermal zone discovery and temperature reading helpers."""

import logging
import sys
from pathlib import Path
from typing import Optional, Set, Union


logger = logging.getLogger(__name__)

KNOWN_GPU_ZONE_TYPES: Set[str] = {
    "GPU-therm",
    "gpu_thermal",
    "Tgpu",
    "BCPU-therm",
    "MCPU-therm",
    "GPU",
    "gpu",
}


def discover_gpu_zone(base_path: Union[str, Path] = "/sys/class/thermal") -> Optional[Path]:
    """Return the temp file path for a known GPU thermal zone, if available."""

    thermal_base = Path(base_path)

    try:
        thermal_zones = sorted(thermal_base.glob("thermal_zone*"))
    except (PermissionError, FileNotFoundError, OSError) as exc:
        logger.warning("Unable to scan thermal zones under %s: %s", thermal_base, exc)
        thermal_zones = []

    for zone_path in thermal_zones:
        try:
            if not zone_path.is_dir():
                continue
        except (PermissionError, FileNotFoundError, OSError) as exc:
            logger.debug("Skipping thermal zone %s: %s", zone_path, exc)
            continue

        type_path = zone_path / "type"
        temp_path = zone_path / "temp"

        try:
            zone_type = type_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.debug("Skipping thermal zone %s because type file is missing.", zone_path)
            continue
        except PermissionError as exc:
            logger.warning("Skipping thermal zone %s; cannot read type file: %s", zone_path, exc)
            continue
        except UnicodeDecodeError as exc:
            logger.warning("Skipping thermal zone %s; type file is not valid UTF-8: %s", zone_path, exc)
            continue
        except OSError as exc:
            logger.warning("Skipping thermal zone %s; cannot read type file: %s", zone_path, exc)
            continue

        if zone_type not in KNOWN_GPU_ZONE_TYPES:
            continue

        try:
            if temp_path.exists():
                logger.info("Found GPU thermal zone %s at %s", zone_path, temp_path)
                return temp_path
        except (PermissionError, FileNotFoundError, OSError) as exc:
            logger.warning("Skipping GPU thermal zone %s; cannot access temp file: %s", zone_path, exc)
            continue

        logger.debug("Skipping GPU thermal zone %s because temp file is missing.", zone_path)

    fallback_path = thermal_base / "thermal_zone1" / "temp"
    try:
        if fallback_path.exists():
            logger.info("Using fallback GPU thermal zone at %s", fallback_path)
            return fallback_path
    except (PermissionError, FileNotFoundError, OSError) as exc:
        logger.warning("Unable to access fallback GPU thermal zone %s: %s", fallback_path, exc)

    return None


def read_temperature(path: Optional[Path]) -> float:
    """Read a sysfs thermal value in millidegrees Celsius as degrees Celsius."""

    if path is None:
        logger.warning("GPU thermal zone path is not available.")
        return 0.0

    temp_path = Path(path)

    try:
        raw_text = temp_path.read_text(encoding="utf-8").strip()
        raw_value = float(raw_text)
    except FileNotFoundError:
        logger.warning("GPU temperature file was not found: %s", temp_path)
        return 0.0
    except PermissionError:
        logger.warning("Permission denied while reading GPU temperature file: %s", temp_path)
        return 0.0
    except ValueError:
        logger.warning("GPU temperature file contains an invalid value: %s", temp_path)
        return 0.0
    except UnicodeDecodeError:
        logger.warning("GPU temperature file is not valid UTF-8: %s", temp_path)
        return 0.0
    except OSError as exc:
        logger.warning("Unable to read GPU temperature file %s: %s", temp_path, exc)
        return 0.0

    if raw_value <= 0:
        logger.warning("Ignoring invalid GPU temperature raw value %s from %s", raw_value, temp_path)
        return 0.0

    return raw_value / 1000.0


def read_gpu_temperature(base_path: Union[str, Path] = "/sys/class/thermal") -> float:
    """Discover and read the GPU temperature in degrees Celsius."""

    return read_temperature(discover_gpu_zone(base_path))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO)
    gpu_zone_path = discover_gpu_zone()
    gpu_temperature = read_temperature(gpu_zone_path)
    print("GPU thermal zone: {}".format(gpu_zone_path))
    print("GPU temperature: {:.1f} \N{DEGREE SIGN}C".format(gpu_temperature))
