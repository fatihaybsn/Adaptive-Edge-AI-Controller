[English](README.md) | [Türkçe](README_TR.md)

# Adaptive Edge-Inference Controller: Thermal-Aware Dynamic Scaling
### Optimized for NVIDIA Jetson Orin NX | TÜBİTAK 2209-A Research Project

This repository contains the development process and experimental results of a hardware-aware control layer designed to prevent thermal throttling and optimize power consumption in Edge AI applications.

## 🚀 Project Overview
High-performance inference on edge devices (like Jetson Orin NX) leads to significant heat generation. Persistent high temperatures cause **Thermal Throttling**, reducing system reliability and lifespan. 

This project implements an **Adaptive Controller** that uses:
- **FOPDT (First-Order Plus Dead Time) Model:** To predict future temperature trends.
- **Fuzzy Logic Controller:** To dynamically adjust **Inference Resolution (imgsz)** and **FPS** limits based on real-time thermal telemetry.

## 📊 Experimental Proof of Concept (PoC)
I conducted a 60-minute continuous inference experiment using YOLOv8 to validate the correlation between resolution and thermal load.

### Experiment Protocol:
- **Duration:** 60 Minutes (Continuous)
- **Phase 1 (0-30m):** Fixed 640px resolution.
- **Phase 2 (30-60m):** Switched to 320px resolution via the control interface (without restarting the system).
- **Environment:** NVIDIA Jetson Orin NX + Raspberry Pi HQ Camera.

![Logo](data/baseline_experiment.png)
![Logo](data/baseline_experiment_GPU.png)
![Logo](data/baseline_experiment_CPU.png)

### Key Findings:
- **Phase 1:** GPU temperature increased by **8°C** within the first 30 minutes.
- **Phase 2:** After switching to 320px, a **3°C drop** was observed within minutes, followed by a stabilized thermal trend.
- **Conclusion:** Dynamic resolution scaling is a viable lever for real-time thermal management.

## Mathematical Background
FOPDT MODEL:
$$G(s) = \frac{K}{Ts + 1} e^{-Ls}$$


## 🛠 Tech Stack
- **Hardware:** NVIDIA Jetson Orin NX
- **Languages:** Python (Async, Typing), C++ (PyBind11 for low-level telemetry)
- **AI Frameworks:** Ultralytics YOLO, TensorRT, CUDA
- **Control Theory:** Fuzzy Logic, FOPDT Modelling
- **Monitoring:** Custom `ThermalGuardianLogger` (CSV-based real-time tracking)

## 📁 Repository Structure
- `src/`: Main source code for inference and control logic.
- `experiments/`: Raw data (`.csv`) and analysis scripts.
- `models/`: Exported TensorRT engine files and weights (placeholders).
- `docs/`: Technical reports and TÜBİTAK submission details.

## 📈 Next Steps
- [ ] Integration of the Fuzzy Logic decision engine with the live inference loop.
- [ ] Implementing sub-10ms state persistence using Redis for multi-agent coordination (LangGraph).
- [ ] Automated Power Mode (NVPModel) switching based on prediction.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.