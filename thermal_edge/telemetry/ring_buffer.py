"""In-memory telemetry ring buffer helpers."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, List, Optional


@dataclass
class TelemetrySample:
    """Single telemetry measurement used by the Adaptive Edge controller."""

    timestamp: float
    gpu_temp: float
    gpu_load: float
    cpu_load: float
    fps: float
    imgsz: int
    percentage: float


class RingBuffer:
    """Fixed-size buffer for telemetry samples."""

    def __init__(self, maxlen: int = 600) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")

        self._buffer: Deque[TelemetrySample] = deque(maxlen=maxlen)

    def push(self, sample: TelemetrySample) -> None:
        if not isinstance(sample, TelemetrySample):
            raise TypeError("sample must be a TelemetrySample")

        self._buffer.append(sample)

    def last(self, n: int = 1) -> List[TelemetrySample]:
        if n <= 0:
            return []

        return list(self._buffer)[-n:]

    def last_one(self) -> Optional[TelemetrySample]:
        if not self._buffer:
            return None

        return self._buffer[-1]

    def last_seconds(
        self, seconds: float, now: Optional[float] = None
    ) -> List[TelemetrySample]:
        if seconds <= 0:
            return []

        current_time = time.monotonic() if now is None else now
        return [
            sample
            for sample in list(self._buffer)
            if current_time - sample.timestamp <= seconds
        ]

    def clear(self) -> None:
        self._buffer.clear()

    def to_list(self) -> List[TelemetrySample]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[TelemetrySample]:
        return iter(list(self._buffer))

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def maxlen(self) -> int:
        return int(self._buffer.maxlen)


if __name__ == "__main__":
    buffer = RingBuffer(maxlen=3)
    start_time = time.monotonic()

    for index in range(4):
        buffer.push(
            TelemetrySample(
                timestamp=start_time + index,
                gpu_temp=50.0 + index,
                gpu_load=10.0 * index,
                cpu_load=15.0 * index,
                fps=30.0,
                imgsz=640,
                percentage=1.0,
            )
        )

    print("RingBuffer length: {}".format(len(buffer)))
    print("Last sample: {}".format(buffer.last_one()))
    print("Recent samples: {}".format(len(buffer.last_seconds(3.0, now=start_time + 4))))
