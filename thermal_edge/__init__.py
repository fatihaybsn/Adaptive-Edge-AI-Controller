"""Adaptive Edge MVP package skeleton."""

__version__ = "0.1.0-alpha"

__all__ = ["__version__", "ControllerConfig", "AdaptiveController"]


def __getattr__(name: str) -> object:
    if name == "ControllerConfig":
        from .config import ControllerConfig

        return ControllerConfig
    if name == "AdaptiveController":
        from .control import AdaptiveController

        return AdaptiveController

    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
