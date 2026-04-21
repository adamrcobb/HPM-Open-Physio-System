#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
Pavlovia-Arduino Bridge with Webcam Recording + GSR Calibration
ENHANCED: 60 FPS Frame-Locked Stimulus Timing + TTL Confirmation
========================================================================

Receives event markers from Pavlovia experiment via WebSocket (Port 5678).
Forwards markers to Arduino and logs synchronized physiological data + video.
Patched for websockets >= 14 / Python 3.13 and to ignore TTL 'T' debug strings.

FIXES v5.1:
FIX 1: VALID_MARKERS list properly closed (was missing closing bracket).
FIX 2: 'CSp_scream.png' corrected (was 'CSpscream.png') to match pipeline STIM_MAP.
FIX 3: experiment_start marker preserved in messages_received after clear(),
so pipeline can compute video-physio time offset correctly.
"""

import asyncio
import websockets
import serial
import serial.tools.list_ports
import datetime
import cv2
import csv
import time
import threading
import traceback
import json
import os
import sys
import numpy as np
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

WEBSOCKET_HOST = 'localhost'
WEBSOCKET_PORT = 5678
ARDUINO_BAUDRATE = 115200

LOG_DIR = "physiologging"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Webcam configuration
WEBCAM_INDEX = 0
TARGET_FPS = 60
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
if sys.platform == 'darwin':
    CODEC = 'mp4v'
else:
    CODEC = 'XVID'
FRAME_INTERVAL_MS = 1000.0 / TARGET_FPS

# GSR Calibration configuration
GSR_CALIBRATION_DURATION_SEC = 30
GSR_SAMPLING_RATE = 249.8

# FIX 1: list properly closed; FIX 2: 'CSp_scream.png' (added underscore)
VALID_MARKERS = [
    'experiment_start', 'experiment_end',
    'baseline_start', 'baseline_end',
    'CSm.png', 'CSp.png', 'CSp_scream.png',
    'GirlScream112dB.wav', 'Silence50ms.wav',
    'calibration_start',
]

START_MARKER = 'experiment_start'
STOP_MARKER = 'experiment_end'
CALIBRATION_START_MARKER = 'calibration_start'

arduino_serial = None
physio_data_file = None
physio_csv_writer = None
recording = False
calibrating = False
messages_received = []
stop_threads = threading.Event()
video_recorder = None

gsr_calibration_data = {
    'baseline_min': None,
    'baseline_max': None,
    'baseline_mean': None,
    'baseline_std': None,
    'baseline_samples_used': 0,
    'calibration_timestamp': None,
}

gsr_calibration_buffer = []

frame_timing_log = []
ttl_confirmations = []

MARKER_PERSIST_MS = 100
current_marker = ""
marker_end_time = 0

# Session tracking
session_physio_csv = None
session_markers_csv = None
session_frame_timing = None
session_video = None
session_timestamp = ""
current_subject = os.environ.get("HPM_SUBJECT_ID", "subject")

# ========================================================================
# FRAME TIMING CLASS
# ========================================================================
class FrameTimer:
    def __init__(self, target_fps=60):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.frame_counter = 0
        self.start_time = None
        self.frame_times = []
        self.last_frame_time = None

    def start(self):
        self.start_time = time.time()
        self.frame_counter = 0
        self.last_frame_time = self.start_time

    def tick(self):
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0.0
        expected_frame = int(elapsed / self.frame_interval)
        if self.last_frame_time:
            self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.frame_counter += 1
        return self.frame_counter, elapsed, expected_frame

    def calculate_jitter(self):
        if not self.frame_times:
            return 0, 0
        mj = sum(self.frame_times) / len(self.frame_times)
        sj = (sum((x - mj) ** 2 for x in self.frame_times) / len(self.frame_times)) ** 0.5
        return mj, sj

# ========================================================================
# VIDEO RECORDING
# ========================================================================
class VideoRecorder(threading.Thread):
    def __init__(self, camera_index, fps, width, height, codec):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.daemon = True
        self.frame_timer = FrameTimer(target_fps=fps)
        self.frame_count = 0

    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print("Failed to open camera!", flush=True)
                return False
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized: {actual_w}x{actual_h} @ {actual_fps}fps (requested {self.fps}, flush=True)")
            print(f"Frame timing: {self.fps} FPS target | {FRAME_INTERVAL_MS:.3f} ms/frame", flush=True)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}", flush=True)
            return False

    def start_recording(self, filename):
        if self.cap is None or not self.cap.isOpened():
            if not self.initialize_camera():
                return False
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
            self.frame_timer.start()
            self.is_recording = True
            global session_video
            session_video = filename
            print(f"▶ Video recording started: {filename}", flush=True)
            return True
        except Exception as e:
            print(f"Error starting recording: {e}", flush=True)
            return False

    def stop_recording(self):
        self.is_recording = False
        time.sleep(0.5)
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"⏹ Total frames recorded: {self.frame_count}", flush=True)
        mj, sj = self.frame_timer.calculate_jitter()
        print(f"Frame jitter — mean: {mj*1000:.3f} ms std: {sj*1000:.3f} ms", flush=True)

    def run(self):
        if not self.initialize_camera():
            return
        fc = 0
        st = None
        while not stop_threads.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if self.is_recording and self.writer:
                    if st is None:
                        st = time.time()
                    self.writer.write(frame)
                    fn, el, ef = self.frame_timer.tick()
                    self.frame_count = fn
                    frame_timing_log.append({
                        'frame_num': fn,
                        'elapsed_sec': el,
                        'expected_frame': ef,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    fc += 1
                    if fc % 100 == 0:
                        cfps = fc / (time.time() - st)
                        print(f"Recording FPS: {cfps:.2f} | Frames captured: {fc}", flush=True)
            except Exception:
                pass
        if self.writer:
            self.writer.release()
        if self.cap:
            self.cap.release()
        print(f"Cleaning up... Total frames recorded: {self.frame_count}", flush=True)

# ========================================================================
# LOGGING UTILITIES
# ========================================================================
def save_frame_timing_log():
    global session_frame_timing
    if not frame_timing_log:
        print("→ No frame timing data collected", flush=True)
        return
    fn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_frametiming.csv")
    try:
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_num', 'elapsed_sec', 'expected_frame', 'timestamp'])
            for e in frame_timing_log:
                w.writerow([e['frame_num'], e['elapsed_sec'], e['expected_frame'], e['timestamp']])
        session_frame_timing = fn
        print(f"✓ Saved frame timing log to {fn}", flush=True)
    except Exception as e:
        print(f"ERROR saving frame timing: {e}", flush=True)

def save_markers_to_csv():
    global session_markers_csv
    if not messages_received:
        return
    fn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_markers.csv")
    try:
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'message', 'client', 'frame_number'])
            for e in messages_received:
                w.writerow([e['timestamp'], e['message'], e['client'], e['frame_number']])
        session_markers_csv = fn
        print(f"✓ Saved event markers to {fn}", flush=True)
    except Exception as e:
        print(f"ERROR saving markers: {e}", flush=True)

def save_ttl_confirmations():
    if not ttl_confirmations:
        return
    fn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_ttl_confirmations.csv")
    try:
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'marker', 'frame_number'])
            for e in ttl_confirmations:
                w.writerow([e['timestamp'], e['marker'], e['frame_number']])
        print(f"✓ Saved TTL confirmations to {fn}", flush=True)
    except Exception as e:
        print(f"ERROR saving TTL confirmations: {e}", flush=True)

# ========================================================================
# ARDUINO COMMUNICATION
# ========================================================================
def find_arduino():
    for p in serial.tools.list_ports.comports():
        if ("Arduino" in p.description or "usbmodem" in p.device or
            (p.vid in [0x2341, 0x1A86, 0x0403, 0x10C4])):
            return p.device
    return None

def initialize_arduino():
    global arduino_serial
    port = find_arduino()
    if port:
        try:
            arduino_serial = serial.Serial(port, ARDUINO_BAUDRATE, timeout=0.1)
            time.sleep(2)
            print(f"✓ Connected to Arduino on {port}", flush=True)
            return True
        except Exception as e:
            print(f"✗ Arduino connection failed: {e}", flush=True)
    print("⚠ No Arduino found. Running in mock mode.", flush=True)
    return False

def initialize_physio_csv_file():
    global physio_data_file, physio_csv_writer, session_physio_csv
    fn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_physiodata.csv")
    try:
        physio_data_file = open(fn, 'w', newline='')
        physio_csv_writer = csv.writer(physio_data_file)
        physio_csv_writer.writerow([
            "PythonSystemTime", "ArduinoSystemTime", "ArduinoTime_ms",
            "RawGSR", "GSR_uS", "RawECG", "ECG_mV", "ArduinoEventMarker", "PythonMarker"
        ])
        session_physio_csv = fn
        print(f"✓ Physio log created: {fn}", flush=True)
        return True
    except Exception as e:
        print(f"Error creating physio log: {e}", flush=True)
        return False

# Matches bare TTL debug lines ('T96456') AND full data lines whose
# Arduino timestamp field starts with T (e.g. 'T96456,96456,...').
# The latter slip through because the comma prevents the bare ^T\d+$ match.
_TTL_DEBUG_RE = re.compile(r'^T\d+([,$]|$)')  # bare 'T96456' OR 'T96456,...'

def read_arduino_data():
    global recording, calibrating, physio_csv_writer, physio_data_file
    global gsr_calibration_buffer, current_marker, marker_end_time

    # ── v5.2.3 fixes ────────────────────────────────────────────────────────
    # FIX 1: Drain ALL waiting serial lines per loop iteration.
    #        The original code read ONE line per loop then slept 1 ms, which
    #        capped throughput well below 250 Hz and caused the pipeline to
    #        auto-detect ~15 Hz. Now we drain the entire serial buffer in a
    #        tight inner while loop before yielding.
    # FIX 2: Arduino v3.3 format has 5 mandatory fields (no duplicate ms col):
    #        [0]ArduinoTime_ms  [1]RawGSR  [2]GSR_uS  [3]RawECG  [4]ECG_mV
    #        [5]eventMarker (optional). Old guard len>=6 rejected every row
    #        without a marker. Now requires len>=5 and maps columns correctly.
    # FIX 3: Calibration GSR index updated: parts[2] = GSR_uS in v3.3.
    # ────────────────────────────────────────────────────────────────────────

    while not stop_threads.is_set():
        try:
            # Drain every line currently in the serial buffer before sleeping
            while arduino_serial and arduino_serial.in_waiting:
                line = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                if line.startswith("TTLPULSESENT,"):
                    try:
                        marker = line.split(",", 1)[1].strip()
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        cf = video_recorder.frame_count if video_recorder else 0
                        ttl_confirmations.append({
                            'timestamp': ts,
                            'marker': marker,
                            'frame_number': cf
                        })
                        print(f"[{ts}] TTL PULSE CONFIRMED: {marker}", flush=True)
                    except Exception as e:
                        print(f"Error parsing TTL confirmation: {e}", flush=True)
                    continue

                # Skip comment/status lines (start with # or known prefixes)
                if any(line.startswith(p) for p in (
                    "#", "=", "Format", "GSR", "Ready", "Send",
                    "Output", "Timestamp", "Arduino", ">"
                )):
                    continue

                # Skip bare TTL debug lines like 'T75', 'T100'
                if _TTL_DEBUG_RE.match(line):
                    continue

                parts = line.split(",")

                # v3.3 format: ArduinoTime_ms,RawGSR,GSR_uS,RawECG,ECG_mV[,marker]
                # Require at least 5 numeric fields
                if len(parts) < 5:
                    continue

                if calibrating:
                    try:
                        # parts[2] = GSR_uS (v3.3 column mapping)
                        gsr_calibration_buffer.append(float(parts[2].strip()))
                        if len(gsr_calibration_buffer) % 250 == 0:
                            print(f" → {len(gsr_calibration_buffer)} GSR samples collected", flush=True)
                    except (ValueError, IndexError):
                        pass

                if recording and physio_csv_writer:
                    py_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    ct = time.time() * 1000
                    pm = current_marker if ct <= marker_end_time else ""
                    # parts[5] is the optional Arduino event marker field
                    arduino_marker = parts[5].strip() if len(parts) >= 6 else ""
                    try:
                        physio_csv_writer.writerow([
                            py_time,            # PythonSystemTime
                            py_time,            # ArduinoSystemTime (Python timestamp; Arduino has no RTC)
                            parts[0].strip(),   # ArduinoTime_ms
                            parts[1].strip(),   # RawGSR
                            parts[2].strip(),   # GSR_uS
                            parts[3].strip(),   # RawECG
                            parts[4].strip(),   # ECG_mV
                            arduino_marker,     # ArduinoEventMarker
                            pm                  # PythonMarker
                        ])
                        physio_data_file.flush()
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(0.001)

# ========================================================================
# GSR CALIBRATION
# ========================================================================
def perform_gsr_calibration():
    global calibrating, gsr_calibration_buffer, gsr_calibration_data
    print("=" * 70, flush=True)
    print("GSR CALIBRATION: Per-Participant Baseline Collection", flush=True)
    print("=" * 70, flush=True)
    print("Calibration starting in 5 seconds...", flush=True)
    for i in range(5, 0, -1):
        print(f"{i}...", flush=True)
        time.sleep(1)
    print("\n▶ CALIBRATION STARTED - PLEASE REMAIN STILL", flush=True)
    gsr_calibration_buffer = []
    calibrating = True

    st = time.time()
    while time.time() - st < GSR_CALIBRATION_DURATION_SEC and not stop_threads.is_set():
        elapsed = int(time.time() - st)
        if elapsed % 5 == 0 and elapsed > 0:
            print(f" {elapsed}s / {GSR_CALIBRATION_DURATION_SEC}s elapsed...", flush=True)
        time.sleep(1)

    calibrating = False
    print("\n⏹ CALIBRATION COMPLETE", flush=True)

    arr = np.array(gsr_calibration_buffer)
    arr = arr[arr > 0]
    if len(arr) > 0:
        gsr_calibration_data['baseline_mean'] = float(np.mean(arr))
        gsr_calibration_data['baseline_std'] = float(np.std(arr))
        gsr_calibration_data['baseline_min'] = float(np.min(arr))
        gsr_calibration_data['baseline_max'] = float(np.max(arr))
        gsr_calibration_data['baseline_samples_used'] = int(len(arr))
        gsr_calibration_data['calibration_timestamp'] = datetime.datetime.now().isoformat()
        print(f"\nCalibration Results:", flush=True)
        print(f" Mean GSR : {gsr_calibration_data['baseline_mean']:.2f} µS", flush=True)
        print(f" Std Dev  : {gsr_calibration_data['baseline_std']:.2f} µS", flush=True)
        print(f" Valid samples: {gsr_calibration_data['baseline_samples_used']}", flush=True)
        fn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_gsr_calibration.json")
        try:
            with open(fn, 'w') as f:
                json.dump(gsr_calibration_data, f, indent=4)
            print(f"✓ GSR calibration saved to {fn}", flush=True)
        except Exception:
            pass
    else:
        print("\n⚠ Calibration failed: No valid GSR samples collected.", flush=True)

def perform_gsr_calibration_threaded():
    threading.Thread(target=perform_gsr_calibration, daemon=True).start()

# ========================================================================
# ANALYSIS PIPELINE LAUNCH
# ========================================================================
def auto_run_pipeline():
    if not session_physio_csv or not session_markers_csv:
        print("⚠ Missing required CSVs. Skipping post-analysis.", flush=True)
        return
    print("\n" + "=" * 70, flush=True)
    print("LAUNCHING POST-ACQUISITION ANALYSIS (v7.17, flush=True)")
    print("=" * 70, flush=True)
    print(f" Physio       : {session_physio_csv}", flush=True)
    print(f" Markers      : {session_markers_csv}", flush=True)
    print(f" Frame timing : {session_frame_timing or 'none'}", flush=True)
    print(f" Video        : {session_video or 'none'}", flush=True)
    try:
        # make sure the filename matches this import
        from psychophysiology_pipeline_v7_17_2 import UnifiedPhysioPipeline, Config
        cfg = Config()
        cfg.PHYSIO_CSV = session_physio_csv
        cfg.MARKERS_CSV = session_markers_csv
        cfg.VIDEO_PATH = session_video
        cfg.FRAME_TIMING_CSV = session_frame_timing
        out_dir = os.path.join(os.path.dirname(session_physio_csv), "analysis_results")
        os.makedirs(out_dir, exist_ok=True)
        cfg.OUTPUT_DIR = out_dir
        UnifiedPhysioPipeline(config=cfg).run()
    except ImportError:
        print("⚠ psychophysiology_pipeline_v7_17_2.py not found or not importable.", flush=True)
    except Exception as e:
        print(f"ERROR in analysis pipeline: {e}", flush=True)
        traceback.print_exc()
    print("=" * 70, flush=True)

# ========================================================================
# WEBSOCKET HANDLER — single-argument form required by websockets >= 14
# ========================================================================
async def handle_client(websocket):
    global recording, calibrating, current_marker, marker_end_time
    global session_timestamp, current_subject

    client_address = websocket.remote_address
    print("=" * 60, flush=True)
    print(f"PAVLOVIA CLIENT CONNECTED from {client_address}", flush=True)
    print("=" * 60, flush=True)

    try:
        async for message in websocket:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            if any(m in message for m in VALID_MARKERS):
                cf = video_recorder.frame_count if video_recorder else 0
                print(f"[{ts}] Frame {cf} → EVENT: {message}", flush=True)

                messages_received.append({
                    'timestamp': ts,
                    'message': message,
                    'client': str(client_address),
                    'frame_number': cf
                })
                current_marker = message
                marker_end_time = time.time() * 1000 + MARKER_PERSIST_MS

                if arduino_serial:
                    arduino_serial.write(f"M,{message}\n".encode('utf-8'))

                if CALIBRATION_START_MARKER in message:
                    perform_gsr_calibration_threaded()

                elif START_MARKER in message:
                    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"\n▶ STARTING EXPERIMENT LOGGING", flush=True)
                    print(f"→ Video frames synced at {FRAME_INTERVAL_MS:.3f} ms/frame", flush=True)
                    if video_recorder:
                        vfn = os.path.join(LOG_DIR, f"{current_subject}_{session_timestamp}_video.mp4")
                        video_recorder.start_recording(vfn)
                    initialize_physio_csv_file()
                    # FIX 3: preserve experiment_start before clearing so pipeline
                    # can compute video-physio time offset from markers CSV
                    start_entry = messages_received[-1]
                    messages_received.clear()
                    frame_timing_log.clear()
                    ttl_confirmations.clear()
                    messages_received.append(start_entry)
                    recording = True

                elif STOP_MARKER in message:
                    print(f"\n⏹ EXPERIMENT END RECEIVED — stopping recording", flush=True)
                    recording = False
                    if video_recorder and video_recorder.is_recording:
                        video_recorder.stop_recording()
                    print("Auto-saving event markers, frame timing, and TTL confirmations...", flush=True)
                    save_markers_to_csv()
                    save_frame_timing_log()
                    save_ttl_confirmations()
                    auto_run_pipeline()

            try:
                await websocket.send(f"Server received: {message}")
            except Exception:
                pass

    except websockets.exceptions.ConnectionClosed:
        print(f"→ Pavlovia client {client_address} disconnected", flush=True)
    except Exception as e:
        print(f"Error in WebSocket handler: {e}", flush=True)
        traceback.print_exc()

# ========================================================================
# MAIN — uses asyncio.run() as required by websockets >= 14
# ========================================================================
async def main():
    print(f"Starting WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}", flush=True)
    print("Waiting for Pavlovia experiment to connect...", flush=True)
    print("System will collect:", flush=True)
    print(" 1. Video recording (1920x1080 @ 60 FPS, flush=True)")
    print(" 2. Physio data — GSR + ECG (physiodata_*.csv, flush=True)")
    print(" 3. Event markers with frame alignment (markers_*.csv, flush=True)")
    print(" 4. Frame timing analysis (frametiming_*.csv, flush=True)")
    print(" 5. GSR calibration (gsr_calibration_*.json, flush=True)")
    print(" 6. TTL pulse confirmations (ttl_confirmations_*.csv, flush=True)")
    print(" 7. Auto-analysis via psychophysiology_pipeline_v7_17_2.py at session end", flush=True)
    print("Ctrl+C to stop", flush=True)

    async with websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("PAVLOVIA-ARDUINO BRIDGE WITH 60 FPS FRAME-LOCKED TIMING", flush=True)
    print("Research-Grade ECG + GSR + Video + Event Markers + TTL Monitoring", flush=True)
    print("=" * 70, flush=True)

    initialize_arduino()

    arduino_thread = threading.Thread(target=read_arduino_data, daemon=True)
    arduino_thread.start()

    video_recorder = VideoRecorder(WEBCAM_INDEX, TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, CODEC)
    video_recorder.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...", flush=True)
        stop_threads.set()

    if video_recorder and video_recorder.is_recording:
        video_recorder.stop_recording()

    if physio_data_file:
        try:
            physio_data_file.close()
        except Exception:
            pass

    if arduino_serial:
        arduino_serial.close()

    save_ttl_confirmations()
    save_markers_to_csv()
    save_frame_timing_log()
    auto_run_pipeline()

    print("✓ Shutdown complete.", flush=True)
    sys.exit(0)
