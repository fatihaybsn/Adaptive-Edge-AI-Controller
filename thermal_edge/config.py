"""Configuration helpers for the Adaptive Edge MVP package."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class ControllerConfig:
    """Runtime configuration for the Adaptive Edge MVP demo."""

    target_temp: float = 65.0
    critical_temp: float = 80.0
    hard_critical_temp: float = 85.0
    control_interval: float = 2.0
    buffer_size: int = 600
    fopdt_time_constant: float = 60.0
    fopdt_dead_time: float = 2.0
    fopdt_prediction_horizon: float = 30.0
    fopdt_max_prediction_delta: float = 10.0
    log_path: str = "adaptive_edge_demo.csv"
    csv_flush_interval: int = 10
    camera_width: int = 960
    camera_height: int = 540
    camera_fps: int = 30
    model_path: str = "insan_tespit_mdeli.pt"
    confidence: float = 0.70
    iou: float = 0.45

    @classmethod
    def from_yaml(cls, path: str) -> "ControllerConfig":
        """Load config from YAML while keeping defaults for missing values."""

        config_path = Path(path)
        if not config_path.exists():
            return cls()

        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with 'pip install PyYAML', or run without a YAML file "
                "to use the default configuration."
            ) from exc

        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}

        if not isinstance(raw_config, Mapping):
            raise ValueError(
                "Config file must contain a YAML mapping with controller, "
                "telemetry, camera, and yolo sections."
            )

        config_values: Dict[str, Any] = {}
        section_fields = {
            "controller": {
                "target_temp": "target_temp",
                "critical_temp": "critical_temp",
                "hard_critical_temp": "hard_critical_temp",
                "control_interval": "control_interval",
                "buffer_size": "buffer_size",
                "fopdt_time_constant": "fopdt_time_constant",
                "fopdt_dead_time": "fopdt_dead_time",
                "fopdt_prediction_horizon": "fopdt_prediction_horizon",
                "fopdt_max_prediction_delta": "fopdt_max_prediction_delta",
            },
            "telemetry": {
                "log_path": "log_path",
                "csv_flush_interval": "csv_flush_interval",
            },
            "camera": {
                "width": "camera_width",
                "height": "camera_height",
                "fps": "camera_fps",
            },
            "yolo": {
                "model_path": "model_path",
                "confidence": "confidence",
                "iou": "iou",
            },
        }
        known_fields = {field.name for field in fields(cls)}

        for section_name, key_map in section_fields.items():
            section_config = raw_config.get(section_name, {})
            if section_config is None:
                continue
            if not isinstance(section_config, Mapping):
                raise ValueError(
                    "Config section '{}' must be a YAML mapping.".format(
                        section_name
                    )
                )

            for yaml_key, field_name in key_map.items():
                if yaml_key in section_config and field_name in known_fields:
                    config_values[field_name] = section_config[yaml_key]

        return cls(**config_values)
