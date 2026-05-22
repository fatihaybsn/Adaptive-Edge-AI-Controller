"""Safety guard for thermal control actions."""

import logging
import time
from typing import Optional, Tuple

from .fuzzy import ControlAction


logger = logging.getLogger(__name__)


class SafetyGuard:
    """Emergency override and recovery guard for control actions."""

    def __init__(
        self,
        target_temp: float = 65.0,
        critical_temp: float = 80.0,
        hard_critical_temp: float = 85.0,
        emergency_imgsz: int = 320,
        emergency_percentage: float = 0.25,
        recovery_margin: float = 5.0,
        recovery_hold_seconds: float = 30.0,
    ) -> None:
        if critical_temp <= target_temp:
            raise ValueError("critical_temp must be greater than target_temp")
        if hard_critical_temp <= critical_temp:
            raise ValueError("hard_critical_temp must be greater than critical_temp")
        if emergency_imgsz <= 0:
            raise ValueError("emergency_imgsz must be positive")
        if emergency_percentage <= 0 or emergency_percentage > 1:
            raise ValueError("emergency_percentage must be in the range (0, 1]")
        if recovery_margin < 0:
            raise ValueError("recovery_margin cannot be negative")
        if recovery_hold_seconds < 0:
            raise ValueError("recovery_hold_seconds cannot be negative")

        self.target_temp = target_temp
        self.critical_temp = critical_temp
        self.hard_critical_temp = hard_critical_temp
        self.emergency_imgsz = emergency_imgsz
        self.emergency_percentage = emergency_percentage
        self.recovery_margin = recovery_margin
        self.recovery_hold_seconds = recovery_hold_seconds
        self._in_emergency = False
        self._recovery_candidate_since: Optional[float] = None

    def determine_mode(self, current_temp: float) -> str:
        if current_temp >= self.hard_critical_temp:
            return "emergency"
        if current_temp >= self.critical_temp:
            return "critical"
        if current_temp >= self.target_temp + 5.0:
            return "warning"
        return "safe"

    def emergency_action(self) -> ControlAction:
        return ControlAction(
            imgsz=self.emergency_imgsz,
            percentage=self.emergency_percentage,
        )

    def validate(
        self,
        action: ControlAction,
        current_temp: float,
        now: Optional[float] = None,
    ) -> Tuple[ControlAction, str]:
        if not isinstance(action, ControlAction):
            raise TypeError("action must be a ControlAction")

        current_time = time.monotonic() if now is None else now

        if current_temp >= self.hard_critical_temp:
            self._in_emergency = True
            self._recovery_candidate_since = None
            logger.critical(
                "Emergency thermal limit reached: %.2f >= %.2f",
                current_temp,
                self.hard_critical_temp,
            )
            return self.emergency_action(), "emergency"

        if self._in_emergency:
            recovery_threshold = self.hard_critical_temp - self.recovery_margin
            if current_temp <= recovery_threshold:
                if self._recovery_candidate_since is None:
                    self._recovery_candidate_since = current_time

                if (
                    current_time - self._recovery_candidate_since
                    >= self.recovery_hold_seconds
                ):
                    self._in_emergency = False
                    self._recovery_candidate_since = None
                else:
                    return self.emergency_action(), "emergency"
            else:
                self._recovery_candidate_since = None
                return self.emergency_action(), "emergency"

        mode = self.determine_mode(current_temp)
        if mode == "emergency":
            return self.emergency_action(), mode

        return action, mode

    def reset(self) -> None:
        self._in_emergency = False
        self._recovery_candidate_since = None


if __name__ == "__main__":
    guard = SafetyGuard()
    action = ControlAction(imgsz=640, percentage=1.0)

    print("Safe: {}".format(guard.validate(action, 60.0)))
    print("Critical: {}".format(guard.validate(action, 82.0)))
    print("Emergency: {}".format(guard.validate(action, 87.0)))
    print("Recovery: {}".format(guard.validate(action, 78.0)))
    guard.reset()
