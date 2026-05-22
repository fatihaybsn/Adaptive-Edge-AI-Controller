"""Windows pre-flight checks for the Adaptive Edge demo script."""

import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORBIDDEN_IMPORTS = ("cv2", "torch", "ultralytics")


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise AssertionError("{}: expected {!r}, got {!r}".format(message, expected, actual))


def assert_contains(lines: Iterable[str], expected: str) -> None:
    assert_condition(
        any(expected in line for line in lines),
        "expected overlay line containing {!r}".format(expected),
    )


def assert_no_forbidden_imports(context: str) -> None:
    imported = [name for name in FORBIDDEN_IMPORTS if name in sys.modules]
    assert_condition(
        not imported,
        "{} imported forbidden modules: {}".format(context, ", ".join(imported)),
    )


def print_process_output(result: subprocess.CompletedProcess) -> None:
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")


def run_command(command: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_import_check() -> Any:
    from examples import basic_yolo_jetson as demo

    assert_no_forbidden_imports("basic_yolo_jetson import")
    print("Import OK")
    return demo


def run_helper_checks(demo: Any) -> None:
    assert_equal(demo.normalize_percentage(0.23), 0.25, "normalize_percentage(0.23)")
    assert_equal(demo.normalize_percentage(0.51), 0.50, "normalize_percentage(0.51)")
    assert_equal(demo.normalize_percentage(0.77), 0.75, "normalize_percentage(0.77)")
    assert_equal(demo.normalize_percentage(0.99), 1.0, "normalize_percentage(0.99)")

    assert_equal(demo.should_run_inference(1, 0.25), True, "frame 1 pct 0.25")
    assert_equal(demo.should_run_inference(2, 0.25), False, "frame 2 pct 0.25")
    assert_equal(demo.should_run_inference(4, 0.25), True, "frame 4 pct 0.25")
    assert_equal(demo.should_run_inference(1, 0.50), True, "frame 1 pct 0.50")
    assert_equal(demo.should_run_inference(2, 0.50), True, "frame 2 pct 0.50")
    assert_equal(demo.should_run_inference(3, 0.50), False, "frame 3 pct 0.50")
    assert_equal(demo.should_run_inference(1, 0.75), True, "frame 1 pct 0.75")
    assert_equal(demo.should_run_inference(4, 0.75), False, "frame 4 pct 0.75")

    estimator = demo.FpsEstimator(window=3)
    estimator.tick(now=100.0)
    estimator.tick(now=101.0)
    estimator.tick(now=102.0)
    assert_condition(abs(estimator.fps - 1.0) < 1e-9, "FpsEstimator fps should be 1.0")

    info = demo.OverlayInfo(
        temp=62.5,
        temp_delta=0.153,
        fps=18.7,
        imgsz=480,
        percentage=0.75,
        mode="warning",
        predicted_temp=64.1,
        control_temp=64.1,
        gpu_load=78.3,
        cpu_load=45.2,
        should_infer=False,
        frame_index=42,
    )
    lines = demo.format_overlay_lines(info)
    assert_contains(lines, "Temp: 62.5 C")
    assert_contains(lines, "Pred: 64.1 C")
    assert_contains(lines, "Ctrl: 64.1 C")
    assert_contains(lines, "Trend: +0.153 C/s")
    assert_contains(lines, "FPS: 18.7")
    assert_contains(lines, "imgsz: 480")
    assert_contains(lines, "pct: 0.75")
    assert_contains(lines, "mode: warning")
    assert_contains(lines, "GPU: 78.3%  CPU: 45.2%")
    assert_contains(lines, "infer: no")
    assert_contains(lines, "frame: 42")

    assert_equal(demo.mode_to_color("safe"), (0, 200, 0), "safe color")
    assert_equal(demo.mode_to_color("warning"), (0, 200, 255), "warning color")
    assert_equal(demo.mode_to_color("critical"), (0, 120, 255), "critical color")
    assert_equal(demo.mode_to_color("emergency"), (0, 0, 255), "emergency color")

    assert_no_forbidden_imports("helper checks")
    print("Helper checks OK")


def run_dry_run_check() -> None:
    command = [sys.executable, "examples/basic_yolo_jetson.py", "--dry-run"]
    result = run_command(command, timeout=10)
    print_process_output(result)

    combined_output = "{}\n{}".format(result.stdout, result.stderr)
    assert_equal(result.returncode, 0, "dry-run return code")
    assert_condition("Dry run OK" in combined_output, "dry-run output should contain 'Dry run OK'")
    print("Dry run OK")


def run_mock_controller_check() -> None:
    script_path = PROJECT_ROOT / "tests" / "manual" / "test_controller_mock_loop.py"
    if not script_path.exists():
        print("WARNING: mock controller loop script not found; skipping")
        return

    command = [sys.executable, str(script_path)]
    result = run_command(command, timeout=10)
    print_process_output(result)

    combined_output = "{}\n{}".format(result.stdout, result.stderr)
    assert_equal(result.returncode, 0, "mock controller loop return code")
    assert_condition("OK" in combined_output, "mock controller loop output should contain 'OK'")
    print("Mock controller loop OK")


def main() -> int:
    print("Adaptive Edge Windows pre-flight")
    try:
        demo = run_import_check()
        run_helper_checks(demo)
        run_dry_run_check()
        run_mock_controller_check()
    except subprocess.TimeoutExpired as exc:
        print("Pre-flight FAILED: command timed out: {}".format(exc.cmd))
        return 1
    except AssertionError as exc:
        print("Pre-flight FAILED: {}".format(exc))
        return 1

    print("Pre-flight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
