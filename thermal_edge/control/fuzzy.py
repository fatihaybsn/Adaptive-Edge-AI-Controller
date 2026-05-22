"""Fuzzy thermal decision logic for the Adaptive Edge MVP."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union


logger = logging.getLogger(__name__)


@dataclass
class ControlAction:
    """Inference-size and frame-percentage decision."""

    imgsz: int
    percentage: float

    def as_dict(self) -> Dict[str, Union[float, int]]:
        return {"imgsz": self.imgsz, "percentage": self.percentage}


class ThermalFuzzyController:
    """Fuzzy controller that maps thermal trend into inference settings."""

    def __init__(
        self,
        target_temp: float = 65.0,
        critical_temp: float = 80.0,
        min_action_interval: float = 10.0,
    ) -> None:
        if critical_temp <= target_temp:
            raise ValueError("critical_temp must be greater than target_temp")
        if min_action_interval < 0:
            raise ValueError("min_action_interval cannot be negative")

        try:
            import numpy as np
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
        except ImportError as exc:
            raise ImportError(
                "scikit-fuzzy is required for ThermalFuzzyController. "
                "Install it with: pip install scikit-fuzzy"
            ) from exc

        self.target_temp = target_temp
        self.critical_temp = critical_temp
        self.min_action_interval = min_action_interval
        self._last_action: Optional[ControlAction] = None
        self._last_action_time: Optional[float] = None
        self._ctrl = ctrl
        self._control_system = self._build_control_system(np, ctrl, fuzz)

    def _build_control_system(self, np: Any, ctrl: Any, fuzz: Any) -> Any:
        temp_error = ctrl.Antecedent(np.linspace(-10, 20, 300), "temp_error")
        temp_delta = ctrl.Antecedent(np.linspace(-2, 2, 300), "temp_delta")
        imgsz_level = ctrl.Consequent(np.linspace(0, 2, 100), "imgsz_level")
        fps_level = ctrl.Consequent(np.linspace(0, 3, 100), "fps_level")

        temp_error["safe"] = fuzz.trimf(temp_error.universe, [-10, -5, 5])
        temp_error["warning"] = fuzz.trimf(temp_error.universe, [3, 10, 17])
        temp_error["critical"] = fuzz.trimf(temp_error.universe, [13, 18, 20])

        temp_delta["falling"] = fuzz.trimf(temp_delta.universe, [-2.0, -1.0, 0.0])
        temp_delta["stable"] = fuzz.trimf(temp_delta.universe, [-0.5, 0.0, 0.5])
        temp_delta["rising"] = fuzz.trimf(temp_delta.universe, [0.2, 1.0, 2.0])

        imgsz_level["low"] = fuzz.trimf(imgsz_level.universe, [0, 0, 1])
        imgsz_level["mid"] = fuzz.trimf(imgsz_level.universe, [0, 1, 2])
        imgsz_level["high"] = fuzz.trimf(imgsz_level.universe, [1, 2, 2])

        fps_level["very_low"] = fuzz.trimf(fps_level.universe, [0, 0, 1])
        fps_level["low"] = fuzz.trimf(fps_level.universe, [0, 1, 2])
        fps_level["mid"] = fuzz.trimf(fps_level.universe, [1, 2, 3])
        fps_level["high"] = fuzz.trimf(fps_level.universe, [2, 3, 3])

        rules = [
            ctrl.Rule(temp_error["safe"] & temp_delta["falling"], (imgsz_level["high"], fps_level["high"])),
            ctrl.Rule(temp_error["safe"] & temp_delta["stable"], (imgsz_level["high"], fps_level["high"])),
            ctrl.Rule(temp_error["safe"] & temp_delta["rising"], (imgsz_level["high"], fps_level["high"])),
            ctrl.Rule(temp_error["warning"] & temp_delta["falling"], (imgsz_level["high"], fps_level["mid"])),
            ctrl.Rule(temp_error["warning"] & temp_delta["stable"], (imgsz_level["high"], fps_level["mid"])),
            ctrl.Rule(temp_error["warning"] & temp_delta["rising"], (imgsz_level["mid"], fps_level["low"])),
            ctrl.Rule(temp_error["critical"] & temp_delta["falling"], (imgsz_level["low"], fps_level["low"])),
            ctrl.Rule(temp_error["critical"] & temp_delta["stable"], (imgsz_level["low"], fps_level["low"])),
            ctrl.Rule(temp_error["critical"] & temp_delta["rising"], (imgsz_level["low"], fps_level["very_low"])),
        ]

        return ctrl.ControlSystem(rules)

    def compute(
        self, current_temp: float, temp_delta: float, now: Optional[float] = None
    ) -> ControlAction:
        current_time = time.monotonic() if now is None else now

        try:
            action = self._compute_action(current_temp, temp_delta)
        except Exception:
            logger.exception("Unexpected error while computing fuzzy control action.")
            if self._last_action is not None:
                return self._last_action
            return ControlAction(imgsz=320, percentage=0.25)

        if self._last_action is None:
            self._last_action = action
            self._last_action_time = current_time
            return action

        if action == self._last_action:
            return self._last_action

        if (
            self._last_action_time is not None
            and current_time - self._last_action_time < self.min_action_interval
        ):
            return self._last_action

        self._last_action = action
        self._last_action_time = current_time
        return action

    def _compute_action(self, current_temp: float, temp_delta: float) -> ControlAction:
        simulation = self._ctrl.ControlSystemSimulation(self._control_system)
        simulation.input["temp_error"] = self._clamp_to_universe(
            current_temp - self.target_temp, -10.0, 20.0
        )
        simulation.input["temp_delta"] = self._clamp_to_universe(temp_delta, -2.0, 2.0)
        simulation.compute()

        imgsz_level = self._nearest_level(simulation.output["imgsz_level"], (0, 1, 2))
        fps_level = self._nearest_level(simulation.output["fps_level"], (0, 1, 2, 3))

        return ControlAction(
            imgsz={0: 320, 1: 480, 2: 640}[imgsz_level],
            percentage={0: 0.25, 1: 0.50, 2: 0.75, 3: 1.00}[fps_level],
        )

    @staticmethod
    def _clamp_to_universe(value: float, minimum: float, maximum: float) -> float:
        clamped = max(minimum, min(maximum, value))
        edge_margin = (maximum - minimum) * 0.01
        if clamped == minimum:
            return minimum + edge_margin
        if clamped == maximum:
            return maximum - edge_margin
        return clamped

    @staticmethod
    def _nearest_level(value: float, levels: Tuple[int, ...]) -> int:
        return min(levels, key=lambda level: abs(value - level))

    def reset(self) -> None:
        self._last_action = None
        self._last_action_time = None


if __name__ == "__main__":
    controller = ThermalFuzzyController(
        target_temp=65.0,
        critical_temp=80.0,
        min_action_interval=0.0,
    )

    print("Safe: {}".format(controller.compute(60.0, -0.5)))
    print("Warning: {}".format(controller.compute(75.0, 0.1)))
    print("Critical: {}".format(controller.compute(82.0, 0.8)))
