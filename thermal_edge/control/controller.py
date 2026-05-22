"""Adaptive Edge controller loop."""

import logging
import threading
import time
from typing import Any, Dict, Optional

from thermal_edge.config import ControllerConfig
from thermal_edge.control.fopdt import FopdtParameters, FopdtThermalPredictor
from thermal_edge.control.fuzzy import ControlAction, ThermalFuzzyController
from thermal_edge.control.safety import SafetyGuard
from thermal_edge.sensors.gpu_load import read_cpu_load, read_gpu_load
from thermal_edge.sensors.thermal_zone import discover_gpu_zone, read_temperature
from thermal_edge.telemetry import CsvLogger, RingBuffer, TelemetrySample


logger = logging.getLogger(__name__)


class AdaptiveController:
    """Background thermal controller for the Adaptive Edge MVP."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self._buffer = RingBuffer(maxlen=config.buffer_size)
        self._fuzzy = ThermalFuzzyController(
            target_temp=config.target_temp,
            critical_temp=config.critical_temp,
        )
        self._fopdt = FopdtThermalPredictor(
            FopdtParameters(
                time_constant_seconds=config.fopdt_time_constant,
                dead_time_seconds=config.fopdt_dead_time,
                prediction_horizon_seconds=config.fopdt_prediction_horizon,
                max_prediction_delta=config.fopdt_max_prediction_delta,
            )
        )
        self._safety = SafetyGuard(
            target_temp=config.target_temp,
            critical_temp=config.critical_temp,
            hard_critical_temp=config.hard_critical_temp,
        )
        self._csv_logger: Optional[CsvLogger] = (
            CsvLogger(config.log_path, flush_interval_rows=config.csv_flush_interval)
            if config.log_path
            else None
        )
        self._zone_path = discover_gpu_zone()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_action: Optional[ControlAction] = ControlAction(
            imgsz=640,
            percentage=1.0,
        )
        self._current_mode = "safe"
        self._latest_temp_delta = 0.0
        self._latest_predicted_temp = 0.0
        self._latest_control_temp = 0.0
        self._latest_fps = 0.0
        self._healthy = True
        self._last_error: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AdaptiveController":
        return cls(ControllerConfig.from_yaml(path))

    def start(self) -> None:
        if self.running:
            return

        self._stop_event.clear()
        with self._lock:
            self._healthy = True
            self._last_error = None

        self._thread = threading.Thread(
            target=self._control_loop,
            name="AdaptiveController",
            daemon=True,
        )
        self._thread.start()
        logger.info("AdaptiveController started")

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

        if self._csv_logger is not None:
            self._csv_logger.close()

        logger.info("AdaptiveController stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def current_action(self) -> ControlAction:
        with self._lock:
            if self._current_action is None:
                return ControlAction(imgsz=640, percentage=1.0)
            return self._current_action

    @property
    def current_mode(self) -> str:
        with self._lock:
            return self._current_mode

    @property
    def latest_temp_delta(self) -> float:
        with self._lock:
            return self._latest_temp_delta

    @property
    def latest_predicted_temp(self) -> float:
        with self._lock:
            return self._latest_predicted_temp

    @property
    def latest_control_temp(self) -> float:
        with self._lock:
            return self._latest_control_temp

    @property
    def latest_fps(self) -> float:
        with self._lock:
            return self._latest_fps

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    def update_fps(self, fps: float) -> None:
        clamped_fps = max(0.0, fps)
        try:
            with self._lock:
                self._latest_fps = clamped_fps
                last_sample = self._buffer.last_one()
                if last_sample is not None:
                    last_sample.fps = clamped_fps
        except Exception as exc:
            logger.warning("Unable to update FPS: %s", exc)

    def _compute_temp_delta(self) -> float:
        try:
            with self._lock:
                samples = self._buffer.last_seconds(30)
            if len(samples) < 4:
                return 0.0

            times = [sample.timestamp for sample in samples]
            temps = [sample.gpu_temp for sample in samples]
            if all(temp == 0.0 for temp in temps):
                return 0.0

            first_time = times[0]
            last_time = times[-1]
            if last_time == first_time:
                return 0.0

            try:
                import numpy as np

                return float(np.polyfit(times, temps, 1)[0])
            except ImportError:
                return (temps[-1] - temps[0]) / (last_time - first_time)
        except Exception as exc:
            logger.warning("Unable to compute temperature delta: %s", exc)
            return 0.0

    def _control_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                temp = read_temperature(self._zone_path)
                gpu_load = read_gpu_load()
                cpu_load = self._read_cpu_load()

                with self._lock:
                    latest_fps = self._latest_fps
                    current_action = self._current_action or ControlAction(
                        imgsz=640,
                        percentage=1.0,
                    )

                sample = TelemetrySample(
                    timestamp=loop_start,
                    gpu_temp=temp,
                    gpu_load=gpu_load,
                    cpu_load=cpu_load,
                    fps=latest_fps,
                    imgsz=current_action.imgsz,
                    percentage=current_action.percentage,
                )
                with self._lock:
                    self._buffer.push(sample)

                temp_delta = self._compute_temp_delta()
                prediction = self._fopdt.predict(temp, temp_delta)
                control_temp = prediction.control_temp
                pre_mode = self._safety.determine_mode(control_temp)
                if pre_mode == "safe":
                    action = ControlAction(imgsz=640, percentage=1.0)
                else:
                    action = self._fuzzy.compute(control_temp, temp_delta)
                action, safety_mode = self._safety.validate(action, temp)
                if safety_mode == "emergency":
                    mode = "emergency"
                elif pre_mode == "emergency":
                    mode = "critical"
                else:
                    mode = pre_mode

                with self._lock:
                    self._current_action = action
                    self._current_mode = mode
                    self._latest_temp_delta = temp_delta
                    self._latest_predicted_temp = prediction.predicted_temp
                    self._latest_control_temp = control_temp

                if self._csv_logger is not None:
                    self._csv_logger.write_row(sample, action, temp_delta, mode)

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0.0, self.config.control_interval - elapsed)
                self._stop_event.wait(sleep_time)
        except Exception as exc:
            logger.exception("AdaptiveController loop failed")
            with self._lock:
                self._healthy = False
                self._last_error = str(exc)
            self._stop_event.set()

    def _read_cpu_load(self) -> float:
        try:
            return read_cpu_load()
        except ImportError:
            return 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "healthy": self.healthy,
            "last_error": self.last_error,
            "current_action": self.current_action.as_dict(),
            "mode": self.current_mode,
            "latest_temp_delta": self.latest_temp_delta,
            "latest_predicted_temp": self.latest_predicted_temp,
            "latest_control_temp": self.latest_control_temp,
            "latest_fps": self.latest_fps,
            "buffer_len": len(self.buffer),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    controller = AdaptiveController(
        ControllerConfig(
            control_interval=0.5,
            log_path="adaptive_edge_controller_smoke.csv",
        )
    )
    controller.start()
    time.sleep(3.0)
    print(controller.status())
    controller.stop()
    print("OK")
