"""Telemetry helpers for the Adaptive Edge MVP package."""

__all__ = ["RingBuffer", "TelemetrySample", "CsvLogger", "CSV_COLUMNS"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    from .ring_buffer import RingBuffer, TelemetrySample
    from .logger import CsvLogger, CSV_COLUMNS

    exports = {
        "RingBuffer": RingBuffer,
        "TelemetrySample": TelemetrySample,
        "CsvLogger": CsvLogger,
        "CSV_COLUMNS": CSV_COLUMNS,
    }
    return exports[name]
