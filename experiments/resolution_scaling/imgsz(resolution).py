# -*- coding: utf-8 -*-
import time
import csv
import os
import threading
import psutil # CPU yükü için: pip install psutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ===== AYARLAR =====
WEIGHTS_PATH = "insan_tespit_mdeli.pt"
CONF_THRES = 0.70
IOU_THRES  = 0.45

# --- GLOBAL DURUM (Logger ve Main arası veri paylaşımı için) ---
class SystemState:
    def __init__(self):
        self.current_res = 640
        self.current_fps = 0.0
        self.gpu_load = 0.0
        self.cpu_load = 0.0

state = SystemState()

# --- PROFESYONEL TERMAL & SISTEM LOGGER ---
class ThermalGuardianLogger(threading.Thread):
    def __init__(self, filename="jetson_detayli_analiz.csv", interval=0.5):
        super().__init__()
        self.filename = filename
        self.interval = interval
        self.running = True
        self.daemon = True
        # Jetson Termal ve GPU Yük Yolları
        self.paths = {
            "GPU_TEMP": "/sys/class/thermal/thermal_zone1/temp",
            "GPU_LOAD": "/sys/devices/gpu.0/load" # Bazı Jetson modellerinde yol değişebilir
        }
        self._setup_file()

    def _setup_file(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # İstediğin sütunlar yan yana
            writer.writerow(["timestamp", "gpu_temperature", "resolution", "fps", "gpu_load", "cpu_load"])

    def get_gpu_temp(self):
        try:
            with open(self.paths["GPU_TEMP"], 'r') as f:
                return float(f.read().strip()) / 1000.0
        except: return 0.0

    def get_gpu_load(self):
        try:
            # Jetson'da GPU yükü 1000 üzerinden verilir (örn: 500 = %50)
            with open(self.paths["GPU_LOAD"], 'r') as f:
                return float(f.read().strip()) / 10.0
        except: return 0.0

    def run(self):
        with open(self.filename, 'a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            while self.running:
                gpu_temp = self.get_gpu_temp()
                state.gpu_load = self.get_gpu_load()
                state.cpu_load = psutil.cpu_percent() # Toplam CPU kullanımı
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Tüm verileri yan yana yaz
                writer.writerow([
                    timestamp, 
                    gpu_temp, 
                    state.current_res, 
                    f"{state.current_fps:.2f}", 
                    state.gpu_load, 
                    state.cpu_load
                ])
                
                f.flush()
                os.fsync(f.fileno())
                time.sleep(self.interval)

    def stop(self):
        self.running = False

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, width=960, height=540, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
    )

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    window_name = "Thermal Controller Panel"
    cv2.namedWindow(window_name)

    # --- ARAYÜZ: Çözünürlük Değiştirme Çubuğu ---
    # 0: 640, 1: 480, 2: 320
    def on_change(val):
        res_map = {0: 640, 1: 480, 2: 320}
        state.current_res = res_map.get(val, 640)
        print(f"[!] Cozunurluk degistirildi: {state.current_res}")

    cv2.createTrackbar("Res (0:640, 1:480, 2:320)", window_name, 0, 2, on_change)

    # Logger'ı başlat
    logger = ThermalGuardianLogger()
    logger.start()

    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            # Anlık seçili olan state.current_res ile inference yap
            results = model.predict(
                source=frame, device=device, imgsz=state.current_res, 
                conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )
            out = results[0].plot()

            # FPS Hesapla ve State'e yaz
            now = time.time()
            dt = now - prev_t
            state.current_fps = 1.0 / dt if dt > 0 else 0
            prev_t = now

            # Ekrana Bilgileri Yazdır
            info = f"RES: {state.current_res} | FPS: {state.current_fps:.1f} | GPU: %{state.gpu_load}"
            cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        logger.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[+] Analiz bitti. Veriler '{logger.filename}' dosyasinda.")

if __name__ == "__main__":
    main()