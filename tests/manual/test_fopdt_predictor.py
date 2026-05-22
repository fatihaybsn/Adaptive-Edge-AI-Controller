"""Manual checks for the FOPDT thermal predictor."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thermal_edge.control.fopdt import FopdtParameters, FopdtThermalPredictor


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    predictor = FopdtThermalPredictor(
        FopdtParameters(
            time_constant_seconds=60.0,
            dead_time_seconds=2.0,
            prediction_horizon_seconds=30.0,
            max_prediction_delta=10.0,
        )
    )

    stable = predictor.predict(current_temp=60.0, temp_delta=0.0)
    assert_condition(stable.predicted_temp == 60.0, "stable prediction should equal current temp")
    assert_condition(stable.control_temp == 60.0, "stable control temp should equal current temp")

    rising = predictor.predict(current_temp=60.0, temp_delta=0.05)
    assert_condition(rising.predicted_temp > 60.0, "rising trend should increase predicted temp")
    assert_condition(rising.control_temp == rising.predicted_temp, "rising control temp should use prediction")

    falling = predictor.predict(current_temp=60.0, temp_delta=-0.05)
    assert_condition(falling.predicted_temp < 60.0, "falling trend should lower predicted temp")
    assert_condition(falling.control_temp == 60.0, "falling control temp should not go below current temp")

    clamped = predictor.predict(current_temp=60.0, temp_delta=10.0)
    assert_condition(clamped.predicted_temp == 70.0, "prediction delta should be clamped")
    assert_condition(clamped.control_temp == 70.0, "clamped control temp should match predicted temp")

    print("FOPDT predictor OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
