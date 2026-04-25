#!/usr/bin/env python3
"""
LIVE PSYCHOPHYSIOLOGY MONITOR WITH REAL-TIME MACHINE LEARNING
Trains an ML model on-the-fly using the ECG as Ground Truth to extract rPPG.
"""

import sys
import time
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import pandas as pd
import threading
from collections import deque
from scipy.signal import butter, sosfiltfilt, find_peaks

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
    import pyqtgraph as pg
    from PyQt5.QtCore import QTimer, Qt
except ImportError:
    print("Please install PyQt5 and pyqtgraph: pip install PyQt5 pyqtgraph pandas")
    sys.exit(1)

try:
    from sklearn.linear_model import Ridge
except ImportError:
    print("Please install scikit-learn for real-time ML: pip install scikit-learn")
    sys.exit(1)

# Configuration
ARDUINO_BAUD_RATE = 115200
WEBCAM_INDEX = 0
FPS_TARGET = 30
DISPLAY_SECONDS = 10  # How many seconds to show on the graphs
MEMORY_SECONDS = 30   # How many seconds the ML algorithm keeps in memory to train on
ECG_FS = 250
RPPG_FS = 30

class LivePhysioMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Physio Monitor: Initializing...")
        self.resize(1200, 900)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Top Numeric BPM Displays ---
        top_layout = QHBoxLayout()
        
        self.ecg_bpm_label = QLabel("ECG BPM: --")
        self.ecg_bpm_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFFF; background-color: #222; padding: 5px; border-radius: 5px;")
        
        self.rppg_bpm_label = QLabel("Standard rPPG: --")
        self.rppg_bpm_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background-color: #222; padding: 5px; border-radius: 5px;")

        self.ml_bpm_label = QLabel("AI Learned rPPG: WAITING")
        self.ml_bpm_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFF00; background-color: #222; padding: 5px; border-radius: 5px;")

        top_layout.addWidget(self.ecg_bpm_label)
        top_layout.addWidget(self.rppg_bpm_label)
        top_layout.addWidget(self.ml_bpm_label)
        layout.addLayout(top_layout)

        # --- Graph Layout ---
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.p1 = self.graph_widget.addPlot(title="Real-Time ECG (Bandpass 0.5-40Hz)")
        self.p1.showGrid(x=True, y=True)
        self.ecg_curve = self.p1.plot(pen=pg.mkPen('c', width=2))
        self.graph_widget.nextRow()

        self.p2 = self.graph_widget.addPlot(title="Real-Time Webcam (Raw Green Channel)")
        self.p2.showGrid(x=True, y=True)
        self.rppg_curve = self.p2.plot(pen=pg.mkPen('g', width=2))
        self.graph_widget.nextRow()

        self.p3 = self.graph_widget.addPlot(title="Heart Rate History (Cyan=ECG, Green=Standard Camera, Yellow=AI Camera)")
        self.p3.setLabel('left', 'BPM')
        self.p3.showGrid(x=True, y=True)
        self.p3.setYRange(40, 150)

        self.hr_ecg_curve = self.p3.plot(pen=pg.mkPen('c', width=2))
        self.hr_rppg_curve = self.p3.plot(pen=pg.mkPen('g', width=2))
        self.hr_ml_curve = self.p3.plot(pen=pg.mkPen('y', width=3))

        # --- Data Buffers ---
        self.ecg_max_len = MEMORY_SECONDS * ECG_FS
        self.ecg_times = deque(maxlen=self.ecg_max_len)
        self.ecg_data = deque(maxlen=self.ecg_max_len)

        self.rppg_max_len = MEMORY_SECONDS * RPPG_FS
        self.vid_times = deque(maxlen=self.rppg_max_len)
        self.vid_r = deque(maxlen=self.rppg_max_len)
        self.vid_g = deque(maxlen=self.rppg_max_len)
        self.vid_b = deque(maxlen=self.rppg_max_len)

        self.ml_times = deque(maxlen=self.rppg_max_len)
        self.ml_hr_preds = deque(maxlen=self.rppg_max_len)

        self.start_time = time.time()
        self.running = True

        # --- ML Model Setup ---
        self.ml_model = Ridge(alpha=100.0)
        self.model_is_trained = False

        # Start background threads
        self.init_arduino_thread()
        self.init_webcam_thread()

        # GUI Update Timer (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33)

        # Machine Learning Training Timer (Trains every 5 seconds)
        self.ml_timer = QTimer()
        self.ml_timer.timeout.connect(self.train_ml_model)
        self.ml_timer.start(5000)

    def init_arduino_thread(self):
        ports = serial.tools.list_ports.comports()
        # Find likely Arduino port (works for macOS and Windows)
        valid_ports = [p for p in ports if any(x in p.description for x in ["Arduino", "USB", "CH340", "Serial"]) 
                       or any(x in p.device for x in ["usbmodem", "usbserial", "cu.usb"])]
        
        port_device = valid_ports[0].device if valid_ports else None
        
        if not port_device and ports:
            # Fallback: Just grab the first port that isn't bluetooth
            non_bt = [p for p in ports if "Bluetooth" not in p.description and "BTH" not in p.device]
            if non_bt: port_device = non_bt[0].device

        self.serial_conn = None
        if port_device:
            try:
                self.setWindowTitle(f"Live Physio Monitor - Connecting to {port_device}...")
                print(f"Connecting to Arduino on {port_device}...")
                self.serial_conn = serial.Serial(port_device, ARDUINO_BAUD_RATE, timeout=0.1)
                
                # CRITICAL: Arduino Unos reset on serial connection. Must wait 2s before writing to it.
                time.sleep(2)
                
                print("Sending 'experiment_start' trigger to Arduino...")
                self.serial_conn.write(b"experiment_start\n")
                self.serial_conn.flush()
                
                self.setWindowTitle(f"Live Physio Monitor - Connected to {port_device}")
                threading.Thread(target=self.read_arduino, daemon=True).start()
            except Exception as e: 
                self.setWindowTitle("Live Physio Monitor - Connection Error!")
                print(f"Could not connect to Arduino: {e}")
        else:
            self.setWindowTitle("Live Physio Monitor - NO ARDUINO FOUND")
            print("No Arduino detected on any port.")

    def read_arduino(self):
        print("Listening for Arduino data stream...")
        first_data = False
        while self.running and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    
                    if not first_data:
                        print(f"Success! First line received: {line}")
                        first_data = True
                        
                    # Ignore header and comments
                    if line.startswith(('#', 'TTL', '=')):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            # Let's extract the ecg_mV robustly by working backwards from the length
                            # Often the ECG mV is the second to last item before the marker (if present), or the last numeric item
                            ecg_val = None
                            for i in range(len(parts)-1, -1, -1):
                                try:
                                    # Try to convert to float. We know ECG is a float like '0.123'
                                    if '.' in parts[i]:
                                        ecg_val = float(parts[i].strip())
                                        break
                                except ValueError:
                                    continue
                            
                            if ecg_val is not None:
                                self.ecg_times.append(time.time() - self.start_time)
                                self.ecg_data.append(ecg_val)
                        except (ValueError, IndexError): 
                            pass
            except Exception as e: 
                time.sleep(0.01)

    def init_webcam_thread(self):
        threading.Thread(target=self.read_webcam, daemon=True).start()

    def read_webcam(self):
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
            return
            
        cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        print("Webcam successfully opened.")

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue

            h, w, _ = frame.shape
            roi = frame[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            
            self.vid_b.append(np.mean(roi[:, :, 0]))
            self.vid_g.append(np.mean(roi[:, :, 1]))
            self.vid_r.append(np.mean(roi[:, :, 2]))
            self.vid_times.append(time.time() - self.start_time)
            
        cap.release()

    def bandpass_filter(self, data, fs, lowcut=0.7, highcut=3.0, order=2):
        if len(data) < 10: return data
        nyq = 0.5 * fs
        if lowcut >= nyq or highcut >= nyq: return data
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
        return sosfiltfilt(sos, data)

    def calculate_hr(self, data, times, fs, is_ecg=False):
        if len(data) < fs * 3: return None, None
        data = np.array(data)
        times = np.array(times)
        
        if is_ecg:
            try: data = self.bandpass_filter(data, fs, lowcut=5.0, highcut=15.0)
            except: data = data - np.mean(data)
            r_peak_approx_height = np.percentile(data, 98)
            threshold = r_peak_approx_height * 0.5 if r_peak_approx_height > 0 else np.std(data) * 1.5
            peaks, _ = find_peaks(data, distance=int(fs * 0.33), height=threshold)
        else:
            try: data = self.bandpass_filter(data, fs, lowcut=0.7, highcut=3.0)
            except: data = data - np.mean(data)
            prom = np.std(data) * 0.25
            peaks, _ = find_peaks(data, distance=int(fs * 0.33), prominence=prom)
            
        if len(peaks) >= 2:
            rr = np.diff(times[peaks])
            valid = (rr >= 0.333) & (rr <= 1.5)
            if not np.any(valid): return None, None
            raw_hr = 60.0 / rr[valid]
            pk_t = times[peaks][1:][valid]
            smoothed_hr = pd.Series(raw_hr).rolling(window=3, min_periods=1).mean().values
            return pk_t, smoothed_hr
        return None, None

    def train_ml_model(self):
        if len(self.ecg_times) < ECG_FS * 10 or len(self.vid_times) < RPPG_FS * 10:
            return

        t_ecg = np.array(self.ecg_times)
        d_ecg = np.array(self.ecg_data)
        min_ecg = min(len(t_ecg), len(d_ecg))
        t_ecg, d_ecg = t_ecg[:min_ecg], d_ecg[:min_ecg]

        t_vid = np.array(self.vid_times)
        r_arr = np.array(self.vid_r)
        g_arr = np.array(self.vid_g)
        b_arr = np.array(self.vid_b)
        min_vid = min(len(t_vid), len(r_arr), len(g_arr), len(b_arr))
        t_vid, r_arr, g_arr, b_arr = t_vid[:min_vid], r_arr[:min_vid], g_arr[:min_vid], b_arr[:min_vid]

        fs_ecg = 1.0 / np.median(np.diff(t_ecg)) if len(t_ecg)>1 else ECG_FS
        t_peaks, hr_ecg = self.calculate_hr(d_ecg, t_ecg, fs_ecg, is_ecg=True)

        if t_peaks is None or len(t_peaks) < 5:
            return

        hr_target = np.interp(t_vid, t_peaks, hr_ecg)

        fs_vid = 1.0 / np.median(np.diff(t_vid)) if len(t_vid)>1 else RPPG_FS
        window_frames = int(3.0 * fs_vid)

        X, y = [], []
        for i in range(window_frames, len(t_vid)):
            r_win = r_arr[i-window_frames:i]
            g_win = g_arr[i-window_frames:i]
            b_win = b_arr[i-window_frames:i]
            
            r_norm = (r_win - np.mean(r_win)) / (np.std(r_win) + 1e-6)
            g_norm = (g_win - np.mean(g_win)) / (np.std(g_win) + 1e-6)
            b_norm = (b_win - np.mean(b_win)) / (np.std(b_win) + 1e-6)
            
            feature = np.concatenate([r_norm, g_norm, b_norm])
            X.append(feature)
            y.append(hr_target[i])

        if len(X) > 0:
            self.ml_model.fit(np.array(X), np.array(y))
            if not self.model_is_trained:
                print("AI Model Training Complete - Predicting Live.")
            self.model_is_trained = True
            self.ml_bpm_label.setText("AI Learned rPPG: ACTIVE")
            self.ml_bpm_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #000; background-color: #FFFF00; padding: 5px; border-radius: 5px;")

    def update_plots(self):
        current_time = time.time() - self.start_time
        display_start = max(0, current_time - DISPLAY_SECONDS)

        self.p1.setXRange(display_start, current_time)
        self.p2.setXRange(display_start, current_time)
        self.p3.setXRange(display_start, current_time)

        ecg_t = np.array(self.ecg_times)
        ecg_d = np.array(self.ecg_data)
        min_ecg = min(len(ecg_t), len(ecg_d))
        ecg_t, ecg_d = ecg_t[:min_ecg], ecg_d[:min_ecg]

        vid_t = np.array(self.vid_times)
        vid_r = np.array(self.vid_r)
        vid_g = np.array(self.vid_g)
        vid_b = np.array(self.vid_b)
        min_vid = min(len(vid_t), len(vid_r), len(vid_g), len(vid_b))
        vid_t, vid_r, vid_g, vid_b = vid_t[:min_vid], vid_r[:min_vid], vid_g[:min_vid], vid_b[:min_vid]

        # 1. Update ECG (Ensures it plots immediately without waiting 2 seconds)
        if len(ecg_t) > 0:
            if len(ecg_d) > ECG_FS * 2:
                try:
                    actual_fs = 1.0 / np.median(np.diff(ecg_t)) if len(ecg_t)>1 else ECG_FS
                    disp_ecg = self.bandpass_filter(ecg_d, actual_fs, lowcut=0.5, highcut=40.0)
                    self.ecg_curve.setData(ecg_t, disp_ecg)
                except: 
                    self.ecg_curve.setData(ecg_t, ecg_d - np.mean(ecg_d))
            else:
                self.ecg_curve.setData(ecg_t, ecg_d - np.mean(ecg_d))

        # 2. Update Standard rPPG Plot
        if len(vid_t) > 0:
            self.rppg_curve.setData(vid_t, vid_g - np.mean(vid_g))

        # 3. Update Standard Heart Rates
        if len(ecg_t) > ECG_FS * 3:
            diffs = np.diff(ecg_t)
            if len(diffs) > 0 and np.median(diffs) > 0:
                pk_t, hr = self.calculate_hr(ecg_d, ecg_t, 1.0/np.median(diffs), is_ecg=True)
                if pk_t is not None and len(hr) > 0:
                    self.hr_ecg_curve.setData(pk_t, hr)
                    recent = hr[pk_t > current_time - 10]
                    if len(recent) > 0: self.ecg_bpm_label.setText(f"ECG BPM: {int(np.median(recent))}")

        if len(vid_t) > RPPG_FS * 3:
            diffs = np.diff(vid_t)
            if len(diffs) > 0 and np.median(diffs) > 0:
                pk_t, hr = self.calculate_hr(vid_g, vid_t, 1.0/np.median(diffs), is_ecg=False)
                if pk_t is not None and len(hr) > 0:
                    self.hr_rppg_curve.setData(pk_t, hr)
                    recent = hr[pk_t > current_time - 10]
                    if len(recent) > 0: self.rppg_bpm_label.setText(f"Standard rPPG: {int(np.median(recent))}")

        # 4. LIVE MACHINE LEARNING PREDICTION
        if self.model_is_trained and len(vid_t) > RPPG_FS * 3.5:
            fs_vid = 1.0 / np.median(np.diff(vid_t))
            window_frames = int(3.0 * fs_vid)

            r_win = vid_r[-window_frames:]
            g_win = vid_g[-window_frames:]
            b_win = vid_b[-window_frames:]

            r_norm = (r_win - np.mean(r_win)) / (np.std(r_win) + 1e-6)
            g_norm = (g_win - np.mean(g_win)) / (np.std(g_win) + 1e-6)
            b_norm = (b_win - np.mean(b_win)) / (np.std(b_win) + 1e-6)

            feature = np.concatenate([r_norm, g_norm, b_norm]).reshape(1, -1)
            predicted_hr = self.ml_model.predict(feature)[0]

            self.ml_times.append(vid_t[-1])
            self.ml_hr_preds.append(predicted_hr)
            
            ml_t = np.array(self.ml_times)
            ml_p = np.array(self.ml_hr_preds)
            min_ml = min(len(ml_t), len(ml_p))

            self.hr_ml_curve.setData(ml_t[:min_ml], ml_p[:min_ml])
            self.ml_bpm_label.setText(f"AI Learned rPPG: {int(predicted_hr)}")

    def closeEvent(self, event):
        self.running = False
        if self.serial_conn: self.serial_conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LivePhysioMonitor()
    window.show()
    sys.exit(app.exec_())
