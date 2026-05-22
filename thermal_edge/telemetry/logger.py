"""CSV telemetry logger for the Adaptive Edge MVP."""

import csv
import datetime
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Type, Union

from thermal_edge.control.fuzzy import ControlAction
from thermal_edge.telemetry.ring_buffer import TelemetrySample


CSV_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "gpu_temp",
    "gpu_load",
    "cpu_load",
    "fps",
    "imgsz",
    "percentage",
    "temp_delta",
    "mode",
)


class CsvLogger:
    """Thread-safe CSV writer for telemetry samples and applied actions."""

    def __init__(self, path: Union[str, Path], flush_interval_rows: int = 10) -> None:
        if isinstance(path, str) and path == "":
            raise ValueError("path cannot be empty")
        if flush_interval_rows <= 0:
            raise ValueError("flush_interval_rows must be a positive integer")

        self.path = Path(path)
        self.flush_interval_rows = flush_interval_rows
        self.path.parent.mkdir(parents=True, exist_ok=True)

        should_write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._lock = threading.Lock()
        self._rows_written = 0
        self._closed = False

        if should_write_header:
            self._writer.writeheader()

    def write_row(
        self,
        sample: TelemetrySample,
        action: ControlAction,
        temp_delta: float,
        mode: str,
    ) -> None:
        if not isinstance(sample, TelemetrySample):
            raise TypeError("sample must be a TelemetrySample")
        if not isinstance(action, ControlAction):
            raise TypeError("action must be a ControlAction")

        with self._lock:
            if self._closed:
                raise ValueError("CsvLogger is closed")

            self._writer.writerow(
                {
                    "timestamp": datetime.datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    "gpu_temp": float(sample.gpu_temp),
                    "gpu_load": float(sample.gpu_load),
                    "cpu_load": float(sample.cpu_load),
                    "fps": float(sample.fps),
                    "imgsz": int(action.imgsz),
                    "percentage": float(action.percentage),
                    "temp_delta": float(temp_delta),
                    "mode": str(mode),
                }
            )
            self._rows_written += 1

            if self._rows_written % self.flush_interval_rows == 0:
                self._file.flush()

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: object,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "adaptive_edge_demo.csv"
        sample = TelemetrySample(
            timestamp=0.0,
            gpu_temp=55.5,
            gpu_load=42.0,
            cpu_load=28.0,
            fps=30.0,
            imgsz=640,
            percentage=1.0,
        )
        action = ControlAction(imgsz=480, percentage=0.5)

        logger = CsvLogger(csv_path, flush_interval_rows=2)
        for _ in range(3):
            logger.write_row(sample, action, temp_delta=0.2, mode="warning")
        logger.close()

        print(csv_path.read_text(encoding="utf-8"))
