# -*- coding: utf-8 -*-
import time
import csv
import os
import threading
import psutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ===== AYARLAR =====
WEIGHTS_PATH = "insan_tespit_mdeli.pt"
CONF_THRES = 0.70
IOU_THRES  = 0.45

class SystemState:
    def __init__(self):
        self.current_res = 640
        self.max_baseline_fps = 0.0 
        self.target_percentage = 1.0 
        self.actual_fps = 0.0
        self.gpu_load = 0.0
        self.cpu_load = 0.0

state = SystemState()

class ThermalGuardianLogger(threading.Thread):
    def __init__(self, filename="jetson_fps_deneyi.csv", interval=0.5):
        super().__init__()
        self.filename = filename
        self.interval = interval
        self.running = True
        self.daemon = True
        self.paths = {
            "GPU_TEMP": "/sys/class/thermal/thermal_zone1/temp", 
            "GPU_LOAD": "/sys/devices/gpu.0/load"
        }
        self._setup_file()

    def _setup_file(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_temp", "resolution", "percentage", "actual_fps", "gpu_load", "cpu_load"])

    def run(self):
        with open(self.filename, 'a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            while self.running:
                try:
                    with open(self.paths["GPU_TEMP"], 'r') as t_file:
                        gpu_temp = float(t_file.read().strip()) / 1000.0
                    with open(self.paths["GPU_LOAD"], 'r') as l_file:
                        state.gpu_load = float(l_file.read().strip()) / 10.0
                except: gpu_temp = 0.0
                
                state.cpu_load = psutil.cpu_percent()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                writer.writerow([timestamp, gpu_temp, state.current_res, state.target_percentage, f"{state.actual_fps:.2f}", state.gpu_load, state.cpu_load])
                f.flush(); os.fsync(f.fileno())
                time.sleep(self.interval)

    def stop(self): self.running = False

def gstreamer_pipeline():
    return ("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, width=960, height=540, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=True")

def main():
    model = YOLO(WEIGHTS_PATH)
    device = "0" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    window_name = "Adaptive Thermal Controller"
    cv2.namedWindow(window_name)

    # --- ADIM 1: KAMERA ACILIS BEKLEME ---
    print("[*] Kamera hatti kuruluyor, ilk goruntu bekleniyor...")
    while True:
        ok, frame = cap.read()
        if ok: break
        time.sleep(0.1)

    # --- ADIM 2: 10 SANIYELIK KALIBRASYON ---
    print("[*] Goruntu alindi. 10 saniyelik gercek zamanli kalibrasyon basliyor...")
    frames, start_calib = 0, time.time()
    while time.time() - start_calib < 10.0:
        ok, frame = cap.read()
        if not ok: break
        model.predict(source=frame, device=device, imgsz=640, verbose=False)
        frames += 1
    state.max_baseline_fps = frames / 10.0
    print(f"[+] Kalibrasyon Tamam. Cihaz Kapasitesi (Baseline): {state.max_baseline_fps:.2f} FPS")

    # --- ARAYÜZ ---
    def on_pct_change(val):
        pct_map = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.25}
        state.target_percentage = pct_map.get(val, 1.0)
        print(f"[!] Hedef Performans: %{state.target_percentage*100}")

    cv2.createTrackbar("Level: 0:%100, 1:%75, 2:%50, 3:%25", window_name, 0, 3, on_pct_change)

    logger = ThermalGuardianLogger()
    logger.start()

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok: break

            results = model.predict(source=frame, device=device, imgsz=state.current_res, verbose=False)
            out = results[0].plot()

            loop_end = time.time()
            state.actual_fps = 1.0 / (loop_end - loop_start)
            
            # Dinamik Limit: Baseline üzerinden hesaplanır
            dynamic_limit = state.max_baseline_fps * state.target_percentage
            desired_period = 1.0 / dynamic_limit
            wait_time = desired_period - (time.time() - loop_start)
            
            if wait_time > 0:
                time.sleep(wait_time)

            info = f"MAX: {state.max_baseline_fps:.1f} | TARGET: %{state.target_percentage*100} | ACT: {state.actual_fps:.1f}"
            cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, out)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        logger.stop(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()