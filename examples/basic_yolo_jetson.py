"""Adaptive Edge MVP YOLO demo entry point."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thermal_edge.config import ControllerConfig
from thermal_edge.control import AdaptiveController


logger = logging.getLogger(__name__)

_VALID_PERCENTAGES = (0.25, 0.50, 0.75, 1.0)


@dataclass
class OverlayInfo:
    temp: float
    temp_delta: float
    fps: float
    imgsz: int
    percentage: float
    mode: str
    predicted_temp: float = 0.0
    control_temp: float = 0.0
    gpu_load: float = 0.0
    cpu_load: float = 0.0
    should_infer: bool = True
    frame_index: int = 0


def format_overlay_lines(info: OverlayInfo) -> list[str]:
    return [
        "Temp: {:.1f} C".format(info.temp),
        "Pred: {:.1f} C".format(info.predicted_temp),
        "Ctrl: {:.1f} C".format(info.control_temp),
        "Trend: {:+.3f} C/s".format(info.temp_delta),
        "FPS: {:.1f}".format(info.fps),
        "imgsz: {}".format(info.imgsz),
        "pct: {:.2f}".format(info.percentage),
        "mode: {}".format(info.mode),
        "GPU: {:.1f}%  CPU: {:.1f}%".format(info.gpu_load, info.cpu_load),
        "infer: {}".format("yes" if info.should_infer else "no"),
        "frame: {}".format(info.frame_index),
    ]


def mode_to_color(mode: str) -> tuple[int, int, int]:
    normalized = str(mode).lower()
    if normalized == "safe":
        return (0, 200, 0)
    if normalized == "warning":
        return (0, 200, 255)
    if normalized == "critical":
        return (0, 120, 255)
    if normalized == "emergency":
        return (0, 0, 255)
    return (255, 255, 255)


class FpsEstimator:
    def __init__(self, window: int = 30) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self, now: Optional[float] = None) -> None:
        self._times.append(time.monotonic() if now is None else now)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0

        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0.0:
            return 0.0

        return (len(self._times) - 1) / elapsed


def normalize_percentage(percentage: float) -> float:
    try:
        value = float(percentage)
    except (TypeError, ValueError):
        return 1.0

    if not math.isfinite(value):
        return 1.0

    value = min(max(value, 0.25), 1.0)
    return min(_VALID_PERCENTAGES, key=lambda allowed: abs(allowed - value))


def should_run_inference(frame_index: int, percentage: float) -> bool:
    if frame_index <= 1:
        return True

    percentage = normalize_percentage(percentage)
    if percentage >= 1.0:
        return True
    if percentage == 0.75:
        return frame_index % 4 != 0
    if percentage == 0.50:
        return frame_index % 2 == 0
    return frame_index % 4 == 0


def gstreamer_pipeline(config: ControllerConfig) -> str:
    return (
        "nvarguscamerasrc sensor_id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate={}/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width={}, height={}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
    ).format(config.camera_fps, config.camera_width, config.camera_height)


def open_camera(cv2: Any, config: ControllerConfig, camera_mode: str, source: str) -> Any:
    if camera_mode == "gstreamer":
        cap = cv2.VideoCapture(gstreamer_pipeline(config), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap

        cap.release()
        raise RuntimeError(
            "Could not open Jetson CSI camera with GStreamer pipeline. "
            "Also verify that Python 3 imports JetPack OpenCV with GStreamer: YES."
        )

    try:
        source_value: Any = int(source)
    except ValueError:
        source_value = source

    cap = cv2.VideoCapture(source_value)
    if cap.isOpened():
        return cap

    cap.release()
    raise RuntimeError("Could not open USB/video camera source")


def wait_for_first_frame(cap: Any, timeout_seconds: float = 10.0) -> object:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok:
            return frame
        time.sleep(0.1)

    raise RuntimeError(
        "Camera opened but no frame was received. Check CSI camera, "
        "nvarguscamerasrc pipeline, and Jetson OpenCV GStreamer support."
    )


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for the demo. Install opencv-python on PC or use Jetson OpenCV build."
        ) from exc

    return cv2


def load_model(model_path: str) -> Tuple[Any, str]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for YOLO inference.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for YOLO inference. Install it with: pip install ultralytics"
        ) from exc

    model = YOLO(model_path)
    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Using YOLO device: %s", device)
    return model, device


def draw_overlay(cv2: Any, frame: Any, info: OverlayInfo) -> Any:
    if frame is None:
        raise ValueError("frame cannot be None")

    lines = ["Adaptive Edge MVP"] + format_overlay_lines(info)
    frame_height, frame_width = frame.shape[:2]
    margin = 12
    x1 = min(margin, max(frame_width - 1, 0))
    y1 = min(margin, max(frame_height - 1, 0))
    panel_width = min(360, max(1, frame_width - x1 - margin))
    line_height = 22
    panel_height = min(20 + line_height * len(lines), max(1, frame_height - y1 - margin))
    x2 = min(frame_width - 1, x1 + panel_width)
    y2 = min(frame_height - 1, y1 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    text_x = x1 + 12
    text_y = y1 + 24
    mode_color = mode_to_color(info.mode)
    for index, line in enumerate(lines):
        if text_y > y2 - 8:
            break

        if index == 0:
            color = mode_color
            font_scale = 0.62
            thickness = 2
        else:
            color = mode_color if line.startswith("mode:") else (230, 230, 230)
            font_scale = 0.52
            thickness = 1

        cv2.putText(
            frame,
            line,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_height

    return frame


def draw_basic_overlay(
    cv2: Any,
    frame: Any,
    *,
    temp: float,
    fps: float,
    imgsz: int,
    percentage: float,
    mode: str,
) -> Any:
    info = OverlayInfo(
        temp=temp,
        temp_delta=0.0,
        fps=fps,
        imgsz=imgsz,
        percentage=percentage,
        mode=mode,
    )
    return draw_overlay(cv2, frame, info)


def run_dry(config_path: str) -> int:
    config = ControllerConfig.from_yaml(config_path)
    controller = AdaptiveController(config)
    controller.start()
    try:
        time.sleep(2.0)
        print(controller.status())
    finally:
        controller.stop()

    print("Dry run OK")
    return 0


def run_demo(args: argparse.Namespace) -> int:
    config = ControllerConfig.from_yaml(args.config)
    cv2 = import_cv2()

    cap = open_camera(cv2, config, args.camera, args.source)
    try:
        print("Camera opened successfully before loading YOLO model")
        print("Waiting for first camera frame...")
        first_frame = wait_for_first_frame(cap, 10.0)
        print("First camera frame received.")
    except Exception:
        cap.release()
        raise

    model, device = load_model(config.model_path)
    cv2.namedWindow(args.window_name)
    ctrl = AdaptiveController(config)
    fps_estimator = FpsEstimator(window=30)
    frame_count = 0
    pending_frame = first_frame
    last_results = None
    last_output = None
    last_inference_fps = 0.0

    ctrl.start()
    try:
        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                frame = pending_frame
                pending_frame = None

            frame_count += 1
            action = ctrl.current_action
            percentage = normalize_percentage(action.percentage)
            should_infer = should_run_inference(frame_count, percentage)
            if should_infer:
                inference_start = time.monotonic()
                results = model.predict(
                    source=frame,
                    device=device,
                    imgsz=action.imgsz,
                    conf=config.confidence,
                    iou=config.iou,
                    verbose=False,
                )
                inference_elapsed = time.monotonic() - inference_start
                last_inference_fps = 1.0 / max(inference_elapsed, 1e-6)
                last_results = results
                out = results[0].plot() if results else frame
                last_output = out.copy()
            else:
                out = last_output.copy() if last_output is not None else frame

            fps_estimator.tick()
            display_fps = fps_estimator.fps
            ctrl.update_fps(display_fps)

            last_sample = ctrl.buffer.last_one()
            temp = last_sample.gpu_temp if last_sample is not None else 0.0
            gpu_load = last_sample.gpu_load if last_sample is not None else 0.0
            cpu_load = last_sample.cpu_load if last_sample is not None else 0.0
            info = OverlayInfo(
                temp=temp,
                temp_delta=ctrl.latest_temp_delta,
                fps=display_fps,
                imgsz=action.imgsz,
                percentage=percentage,
                mode=ctrl.current_mode,
                predicted_temp=ctrl.latest_predicted_temp,
                control_temp=ctrl.latest_control_temp,
                gpu_load=gpu_load,
                cpu_load=cpu_load,
                should_infer=should_infer,
                frame_index=frame_count,
            )
            out = draw_overlay(cv2, out, info)

            cv2.imshow(args.window_name, out)
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        ctrl.stop()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Edge MVP YOLO demo")
    parser.add_argument("--config", default="examples/configs/default.yaml")
    parser.add_argument(
        "--camera",
        choices=["gstreamer", "usb"],
        default="gstreamer",
        help="Camera backend. Use gstreamer for Jetson CSI/Flex/Raspberry Pi HQ camera.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="USB camera index or video path. Only used for --camera usb or fallback.",
    )
    parser.add_argument("--window-name", default="Adaptive Edge MVP")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry(args.config)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
