"""FOPDT-based thermal prediction helpers."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FopdtParameters:
    time_constant_seconds: float = 60.0
    dead_time_seconds: float = 2.0
    prediction_horizon_seconds: float = 30.0
    max_prediction_delta: float = 10.0

    def __post_init__(self) -> None:
        if self.time_constant_seconds <= 0:
            raise ValueError("time_constant_seconds must be positive")
        if self.dead_time_seconds < 0:
            raise ValueError("dead_time_seconds cannot be negative")
        if self.prediction_horizon_seconds < 0:
            raise ValueError("prediction_horizon_seconds cannot be negative")
        if self.max_prediction_delta <= 0:
            raise ValueError("max_prediction_delta must be positive")


@dataclass(frozen=True)
class FopdtPrediction:
    predicted_temp: float
    control_temp: float
    effective_horizon_seconds: float


class FopdtThermalPredictor:
    def __init__(self, parameters: FopdtParameters) -> None:
        self.parameters = parameters

    def predict(self, current_temp: float, temp_delta: float) -> FopdtPrediction:
        effective_horizon = max(
            0.0,
            self.parameters.prediction_horizon_seconds - self.parameters.dead_time_seconds,
        )
        if effective_horizon == 0.0 or temp_delta == 0.0:
            predicted_temp = current_temp
        else:
            response_ratio = 1.0 - math.exp(
                -effective_horizon / self.parameters.time_constant_seconds
            )
            raw_delta = self.parameters.time_constant_seconds * temp_delta * response_ratio
            prediction_delta = self._clamp(
                raw_delta,
                -self.parameters.max_prediction_delta,
                self.parameters.max_prediction_delta,
            )
            predicted_temp = current_temp + prediction_delta

        return FopdtPrediction(
            predicted_temp=predicted_temp,
            control_temp=max(current_temp, predicted_temp),
            effective_horizon_seconds=effective_horizon,
        )

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
