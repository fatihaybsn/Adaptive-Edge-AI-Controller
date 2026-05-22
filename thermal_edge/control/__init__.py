"""Control helpers for the Adaptive Edge MVP package."""

__all__ = [
    "ControlAction",
    "ThermalFuzzyController",
    "SafetyGuard",
    "FopdtParameters",
    "FopdtPrediction",
    "FopdtThermalPredictor",
    "AdaptiveController",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    from .fopdt import FopdtParameters, FopdtPrediction, FopdtThermalPredictor
    from .fuzzy import ControlAction, ThermalFuzzyController
    from .safety import SafetyGuard
    from .controller import AdaptiveController

    exports = {
        "ControlAction": ControlAction,
        "ThermalFuzzyController": ThermalFuzzyController,
        "SafetyGuard": SafetyGuard,
        "FopdtParameters": FopdtParameters,
        "FopdtPrediction": FopdtPrediction,
        "FopdtThermalPredictor": FopdtThermalPredictor,
        "AdaptiveController": AdaptiveController,
    }
    return exports[name]
