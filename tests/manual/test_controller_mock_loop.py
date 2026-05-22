"""Manual mock-loop check for AdaptiveController."""

import csv
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Sequence
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thermal_edge.config import ControllerConfig
from thermal_edge.control.controller import AdaptiveController


def sequence_reader(values: Sequence[float]) -> Callable[..., float]:
    index = 0

    def read_next(*args: object, **kwargs: object) -> float:
        nonlocal index
        if index < len(values):
            value = values[index]
            index += 1
            return value
        return values[-1]

    return read_next


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_mock_loop() -> None:
    temps = [
        50.0,
        52.0,
        55.0,
        58.0,
        62.0,
        66.0,
        70.0,
        74.0,
        79.0,
        82.0,
        81.0,
        78.0,
        74.0,
        72.0,
        68.0,
    ]
    gpu_loads = [40.0, 55.0, 70.0, 85.0, 90.0]

    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "mock_controller.csv"
        config = ControllerConfig(
            target_temp=55.0,
            critical_temp=70.0,
            hard_critical_temp=80.0,
            control_interval=0.1,
            buffer_size=200,
            log_path=str(csv_path),
            csv_flush_interval=2,
        )

        with patch("thermal_edge.control.controller.discover_gpu_zone", return_value=None), patch(
            "thermal_edge.control.controller.read_temperature",
            side_effect=sequence_reader(temps),
        ), patch(
            "thermal_edge.control.controller.read_gpu_load",
            side_effect=sequence_reader(gpu_loads),
        ), patch(
            "thermal_edge.control.controller.read_cpu_load",
            return_value=25.0,
        ):
            ctrl = AdaptiveController(config)
            print("Mock controller loop test", flush=True)
            ctrl.start()
            try:
                time.sleep(0.6)
                ctrl.update_fps(18.0)
                time.sleep(0.5)
                ctrl.update_fps(17.5)
                time.sleep(0.5)
                ctrl.update_fps(16.0)
                time.sleep(0.4)
                status = ctrl.status()
                print("Status: {}".format(status), flush=True)
            finally:
                ctrl.stop()

            assert_condition(ctrl.healthy, "controller should remain healthy")
            assert_condition(not ctrl.running, "controller should not be running after stop")
            assert_condition(csv_path.exists(), "CSV log file was not created")
            assert_condition(
                status["buffer_len"] >= 8,
                "buffer length should be at least 8, got {}".format(status["buffer_len"]),
            )
            assert_condition(
                "latest_predicted_temp" in status,
                "status should include latest_predicted_temp",
            )
            assert_condition(
                "latest_control_temp" in status,
                "status should include latest_control_temp",
            )
            assert_condition(
                status["latest_control_temp"] >= status["latest_predicted_temp"]
                or status["latest_control_temp"] >= 0.0,
                "status should include a valid control temperature",
            )

        rows = read_csv_rows(csv_path)
        modes_seen = sorted({row["mode"] for row in rows})
        actions_seen = sorted(
            {
                (int(row["imgsz"]), float(row["percentage"]))
                for row in rows
            }
        )

        print("CSV rows: {}".format(len(rows)), flush=True)
        print("Modes seen: {}".format(modes_seen), flush=True)
        print("Actions seen: {}".format(actions_seen), flush=True)

        assert_condition(len(rows) >= 8, "expected at least 8 CSV rows, got {}".format(len(rows)))
        assert_condition(
            "emergency" in modes_seen,
            "expected emergency mode in CSV rows, got {}".format(modes_seen),
        )
        assert_condition(
            (320, 0.25) in actions_seen,
            "expected emergency action (320, 0.25), got {}".format(actions_seen),
        )

    print("OK", flush=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    run_mock_loop()
