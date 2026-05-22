# Adaptive Edge-Inference Controller

**Thermal-aware dynamic scaling for YOLO inference on NVIDIA Jetson-class edge AI devices.**

This project implements an application-level control layer that monitors GPU temperature, predicts near-future thermal pressure, and dynamically adjusts YOLO inference workload before sustained heat causes throttling, unstable latency, frame drops, freezes, or shutdowns.

<p align="center">
  <img src="images/Demo-thermal.png" alt="Adaptive Edge-Inference Controller live thermal demo" width="900">
</p>

<p align="center"><em>Live demonstration overlay: measured temperature, predicted/control temperature, FPS, GPU/CPU load, operating mode, inference ratio, and YOLO input size.</em></p>

## Key demo outcomes

The closed-loop demonstration shows that the controller can operate for a long continuous run while adapting inference settings across thermal regions.

| Result | Value |
| --- | ---: |
| Closed-loop demo duration | ~130 minutes |
| Logged telemetry samples | 3,900 |
| Maximum GPU temperature | 81.815 °C |
| First 70 °C crossing | ~22.101 min |
| First 80 °C crossing | ~72.335 min |
| 85 °C hard emergency threshold exceeded? | No |
| Emergency mode triggered in shared demo? | No |

The system enters warning and critical thermal regions, but the shared run does not cross the 85 °C hard emergency threshold.

## Why this matters

Real-time computer vision workloads on embedded edge devices are often thermally constrained. A YOLO pipeline may run correctly at startup, then degrade after minutes or hours as heat accumulates. In field deployments this can appear as:

- thermal throttling and unstable FPS,
- latency accumulation in video pipelines,
- dropped frames or reduced detection responsiveness,
- process freezes that require manual restart,
- thermal or power-related shutdowns.

This is especially relevant for unattended Jetson deployments such as smart cameras, traffic analytics nodes, mobile robots, UAVs, industrial monitoring devices, and outdoor AI sensors.

Adaptive Edge-Inference Controller addresses this by adding a software control layer above the inference loop. Instead of waiting for the platform to throttle, the application reduces workload proactively and restores performance when thermal conditions improve.

## What the project does

The controller combines real-time telemetry, short-horizon thermal prediction, fuzzy decision logic, and a safety guard:

1. Read Jetson GPU temperature, GPU load, CPU load, and live FPS.
2. Store recent telemetry in a ring buffer.
3. Estimate the current thermal trend.
4. Predict near-future control temperature with an FOPDT-inspired predictor.
5. Use fuzzy logic to choose an inference workload.
6. Apply a safety guard for critical and emergency thermal states.
7. Log every control step to CSV for analysis and reproducibility.

The controller changes two inference workload levers:

| Lever | Meaning | Effect |
| --- | --- | --- |
| `imgsz` | YOLO input image size | Reduces per-frame compute cost |
| `percentage` | Fraction of frames sent to inference | Reduces inference frequency / effective workload |

## Operating modes

| Mode | Approximate condition | Behavior |
| --- | --- | --- |
| `safe` | Below warning region | Full-quality inference: `imgsz=640`, `percentage=1.0` |
| `warning` | Around 70 °C and above | Gradually reduce workload through fuzzy control |
| `critical` | Around 80 °C and above | Stronger reduction through fuzzy control |
| `emergency` | 85 °C hard limit | Force emergency action: `imgsz=320`, `percentage=0.25` |

The default thresholds are defined in [`examples/configs/default.yaml`](examples/configs/default.yaml).

## System architecture

<p align="center">
  <img src="images/system-architecture.png" alt="Adaptive Edge-Inference Controller system architecture" width="900">
</p>

The runtime path is implemented as:

```text
examples/basic_yolo_jetson.py
  -> thermal_edge.config.ControllerConfig
  -> thermal_edge.control.controller.AdaptiveController
  -> thermal_edge.sensors.thermal_zone / gpu_load
  -> thermal_edge.telemetry.RingBuffer
  -> thermal_edge.control.fopdt.FopdtThermalPredictor
  -> thermal_edge.control.fuzzy.ThermalFuzzyController
  -> thermal_edge.control.safety.SafetyGuard
  -> thermal_edge.telemetry.CsvLogger
```

## Closed-loop control mechanism

<p align="center">
  <img src="images/closed-Loop-Control-Mechanism.png" alt="Closed-loop thermal control mechanism" width="850">
</p>

The inference loop reads the latest controller action before each frame. The controller runs in a background thread, updates telemetry periodically, and publishes the current `imgsz` and `percentage` decision.

## Technologies and layers

<p align="center">
  <img src="images/Technologies-and-Layers.png" alt="Technologies and layers" width="900">
</p>

Codebase-verified technologies include:

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- scikit-fuzzy
- PyYAML
- psutil
- Jetson sysfs thermal/load telemetry
- GStreamer camera pipeline for Jetson CSI cameras

PyTorch, OpenCV, CUDA, and camera support on Jetson depend on the JetPack version and installed platform packages.

## Experimental results

### Closed-loop adaptive demo

Source data: [`experiments/closed_loop_demo/adaptive_edge_demo.csv`](experiments/closed_loop_demo/adaptive_edge_demo.csv)

<p align="center">
  <img src="images/demonstration_experiment_gpu_temperature_from_csv_EN.png" alt="Closed-loop GPU temperature over time" width="900">
</p>

| Metric | Value |
| --- | ---: |
| Samples | 3,900 |
| Duration | ~129.97 min |
| GPU temperature range | 52.562 °C to 81.815 °C |
| Average GPU temperature | 74.1666 °C |
| First 70 °C crossing | ~22.101 min |
| First 80 °C crossing | ~72.335 min |
| Share of run above 80 °C | 10.0% |
| 85 °C exceeded | No |

<p align="center">
  <img src="images/demo_experiment_operating_mode_distribution_EN.png" alt="Closed-loop operating mode distribution" width="650">
</p>

| Mode | Samples | Share |
| --- | ---: | ---: |
| `safe` | 726 | 18.6% |
| `warning` | 2,784 | 71.4% |
| `critical` | 390 | 10.0% |

The demo shows a sustained adaptive run rather than a short startup snapshot: the system moves through safe, warning, and critical regions while keeping the shared run below the hard emergency threshold.

### FPS percentage experiment

Source data: [`experiments/fps_percentage/fps.csv`](experiments/fps_percentage/fps.csv)

This experiment changes the processed-frame percentage from `1.0` to `0.25` while keeping resolution fixed at 640.

| Phase | Avg GPU temp | Avg FPS | Avg GPU load | Avg CPU load |
| --- | ---: | ---: | ---: | ---: |
| `percentage=1.0` | 56.47 °C | 24.06 | 72.88% | 19.10% |
| `percentage=0.25` | 54.63 °C | 8.34 | 26.39% | 10.70% |

Takeaway: frame-percentage reduction is the strongest thermal actuator in the shared experiments, but it has an explicit throughput cost.

Additional plots are available in:

- [`experiments/fps_percentage/fps_thermal.png`](experiments/fps_percentage/fps_thermal.png)
- [`experiments/fps_percentage/fps_gpu_load.png`](experiments/fps_percentage/fps_gpu_load.png)
- [`experiments/fps_percentage/fps_load_CPU.png`](experiments/fps_percentage/fps_load_CPU.png)

### Resolution scaling experiment

Source data: [`experiments/resolution_scaling/resolution.csv`](experiments/resolution_scaling/resolution.csv)

This experiment changes YOLO input size from 640 to 320.

| Phase | Avg GPU temp | Avg FPS | Avg GPU load | Avg CPU load |
| --- | ---: | ---: | ---: | ---: |
| `resolution=640` | 57.15 °C | 23.75 | 71.03% | 18.98% |
| `resolution=320` | 55.99 °C | 25.87 | 67.04% | 19.11% |

Takeaway: reducing `imgsz` gives a milder thermal reduction than frame-percentage limiting, but it can preserve or even improve throughput by lowering per-frame compute cost.

Additional plots are available in:

- [`experiments/resolution_scaling/resolution_thermal.png`](experiments/resolution_scaling/resolution_thermal.png)
- [`experiments/resolution_scaling/resolution_load-GPU.png`](experiments/resolution_scaling/resolution_load-GPU.png)
- [`experiments/resolution_scaling/resolution_load-CPU.png`](experiments/resolution_scaling/resolution_load-CPU.png)

### Offline FOPDT identification

Offline identification results are stored in [`thermal_edge/control/fopdt_fit_results.json`](thermal_edge/control/fopdt_fit_results.json).

| Step experiment | K | Tau | Theta | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPS percentage step | 3.96 °C/u | 119.1 s | 0.0 s | 0.154 °C | 0.936 |
| Image-size step | 2.50 °C/u | 82.3 s | 5.2 s | 0.076 °C | 0.963 |

Important: these are offline identification results. The current runtime predictor is FOPDT-inspired and uses configured parameters from [`examples/configs/default.yaml`](examples/configs/default.yaml); it does not automatically load `fopdt_fit_results.json`.

## Use cases

This project is relevant when an edge AI workload must remain stable for long-running deployments under constrained cooling:

- smart surveillance and security cameras,
- human, vehicle, or license-plate detection at the edge,
- traffic and parking analytics,
- mobile robots and UAVs,
- industrial safety monitoring,
- outdoor sensor nodes,
- unattended remote Jetson deployments.

The repository includes example model weights under [`models/`](models/), but no accuracy benchmark is claimed in this README.

## Repository structure

```text
.
├── docs/
│   └── Adaptive_Edge_AI_Controller_Report.pdf
├── examples/
│   ├── basic_yolo_jetson.py
│   └── configs/default.yaml
├── experiments/
│   ├── closed_loop_demo/
│   ├── fps_percentage/
│   └── resolution_scaling/
├── images/
│   └── README figures and result visuals
├── models/
│   └── sample .pt model weights
├── tests/manual/
│   └── manual validation scripts
└── thermal_edge/
    ├── config.py
    ├── control/
    ├── sensors/
    └── telemetry/
```

## Setup

This repository includes a Jetson-focused [`requirements.txt`](requirements.txt). It intentionally avoids pip OpenCV, generic pip PyTorch, and generic pip torchvision because those packages can replace JetPack-compatible builds on Jetson.

On Jetson, create an environment that can see JetPack system packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

On non-Jetson systems, a regular environment is fine:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Jetson, install PyTorch and torchvision using NVIDIA/JetPack-compatible wheels or packages for your exact JetPack/CUDA version. Do not install generic pip `torch` or `torchvision` unless you have verified that those wheels are compatible with your Jetson image.

Install the pip-safe Python dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install Ultralytics without dependency resolution so pip does not pull `opencv-python`:

```bash
python3 -m pip install --no-deps ultralytics==8.4.53 ultralytics-thop==2.0.18
```

### Jetson OpenCV and GStreamer warning

If the camera is opened through GStreamer, OpenCV must be built with GStreamer support. On Jetson this should normally be the OpenCV build that comes with JetPack/L4T. Do not install `opencv-python`, `opencv-contrib-python`, or `opencv-python-headless` from pip for the Jetson GStreamer camera path, because those wheels can shadow the JetPack OpenCV build and break `cv2.CAP_GSTREAMER`.

If a pip OpenCV package is already installed, remove it:

```bash
python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
```

After removing pip OpenCV, Python should fall back to the JetPack-provided `cv2` module when the environment can access system site packages. This is also the expected OpenCV build for the Jetson GPU/GStreamer camera workflow used by this project.

Verify the active OpenCV build:

```bash
python3 - <<'PY'
import cv2

print(cv2.__file__)
print(cv2.__version__)
for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line:
        print(line)
PY
```

## Configuration

Default configuration lives at [`examples/configs/default.yaml`](examples/configs/default.yaml).

Important fields:

| Field | Purpose |
| --- | --- |
| `controller.target_temp` | Desired thermal operating target |
| `controller.critical_temp` | Critical control threshold |
| `controller.hard_critical_temp` | Emergency threshold |
| `controller.control_interval` | Background controller interval |
| `telemetry.log_path` | CSV output path |
| `camera.width`, `camera.height`, `camera.fps` | Camera pipeline settings |
| `yolo.model_path` | YOLO model path |
| `yolo.confidence`, `yolo.iou` | Detection thresholds |

Before running the live demo, update `yolo.model_path` to an existing model file. The current config points to `insan_tespit_mdeli.pt`, while this repository currently includes model files such as:

- [`models/best_human.pt`](models/best_human.pt)
- [`models/best_car.pt`](models/best_car.pt)
- [`models/best_plaka_plate.pt`](models/best_plaka_plate.pt)

Example:

```yaml
yolo:
  model_path: "models/best_human.pt"
  confidence: 0.70
  iou: 0.45
```

## Quick start

Run a controller dry run without opening the camera or loading YOLO:

```bash
python examples/basic_yolo_jetson.py --dry-run
```

Run the default Jetson CSI/GStreamer demo:

```bash
python examples/basic_yolo_jetson.py --config examples/configs/default.yaml
```

Run with a USB camera or video source:

```bash
python examples/basic_yolo_jetson.py --camera usb --source 0
```

Run for a bounded number of frames:

```bash
python examples/basic_yolo_jetson.py --max-frames 300
```

Press `q` in the OpenCV window to quit the live demo.

## Manual checks

```bash
python tests/manual/test_fopdt_predictor.py
python tests/manual/test_controller_mock_loop.py
python tests/manual/test_demo_preflight_windows.py
```

These are manual validation scripts rather than a full CI test suite.

## Runtime output

The live overlay shows:

- measured GPU temperature,
- predicted temperature,
- control temperature,
- temperature trend,
- display FPS,
- selected `imgsz`,
- selected `percentage`,
- operating mode,
- GPU and CPU load,
- whether the current frame is inferred,
- frame index.

The CSV logger writes:

```text
timestamp,gpu_temp,gpu_load,cpu_load,fps,imgsz,percentage,temp_delta,mode
```

## Full technical report

The full project report is available at:

- [`docs/Adaptive_Edge_AI_Controller_Report.pdf`](docs/Adaptive_Edge_AI_Controller_Report.pdf)

It contains the broader motivation, experimental discussion, figures, and references used to contextualize the project.

## Field evidence and further reading

Community reports and platform documentation referenced by the project include:

- [YOLOv8 on Jetson Nano reaching high temperatures](https://forums.developer.nvidia.com/t/yolov8-on-jetson-nano/265650)
- [Jetson Nano overheating and shutdown discussion](https://forums.developer.nvidia.com/t/overheating-shut-down-jetson-nano/80693)
- [Heavy YOLO inferencing freeze discussion](https://community.ultralytics.com/t/system-freeze-when-performing-heavy-yolo-inferencing/1882)
- [DeepStream nvinfer delay and latency accumulation](https://forums.developer.nvidia.com/t/delay-due-to-nvinfer/324632)
- [Jetson inference high temperature / shutdown issue](https://github.com/dusty-nv/jetson-inference/issues/1473)
- [Long-running Jetson Nano overheating concern](https://forums.developer.nvidia.com/t/jetson-nano-long-run-over-heat/157545)
- [detectnet-camera crash and 5W mode discussion](https://forums.developer.nvidia.com/t/jetson-nano-crashing-while-using-detectnet-camera-demo-from-jetson-inference/74384)
- [TensorFlow power / over-current throttling discussion](https://forums.developer.nvidia.com/t/power-error-while-using-tensorflow/181327)
- [Jetson Nano freeze while detecting](https://forums.developer.nvidia.com/t/jetson-nano-freezes-while-detecting/214700)
- [DeepStream thermal throttling on Jetson Orin Nano Super](https://forums.developer.nvidia.com/t/deepstream-7-1-on-jetson-orin-nano-super-3-stream-pipeline-thermal-throttle-at-68-70-c-seeking-fps-optimization-advice/364742)
- [NVIDIA Jetson Orin Nano Developer Kit User Guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/index.html)
- [NVIDIA Jetson tegrastats utility](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [Jetson power and thermal management documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

## Known limitations

- Jetson setup depends on JetPack-compatible OpenCV, PyTorch, and torchvision builds; `requirements.txt` deliberately avoids replacing those platform packages.
- The default `yolo.model_path` must be changed before running the live demo with the current repository contents.
- Jetson telemetry depends on sysfs paths that may vary across JetPack versions and board configurations.
- The default camera backend is Jetson CSI/GStreamer; desktop systems may need `--camera usb --source ...`.
- Offline FOPDT identification results are not automatically loaded by the current runtime predictor.
- The shared experiments evaluate thermal/performance behavior, not detection accuracy.
- Lowering `percentage` or `imgsz` trades thermal relief against temporal coverage, throughput, or detection quality.
- Emergency mode is implemented, but the shared closed-loop demo did not exceed the 85 °C emergency threshold.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
