#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPM System GUI
==============================================================================
Tkinter-based launcher and session monitor for the HPM psychophysiology system.

Two modes:
  • RA Mode         – full control panel for research assistants
  • Participant Mode – simplified step-by-step wizard for remote/home use

Requires (same folder):
  pavlovia_arduino_bridge_v5_2_2.py
  psychophysiology_pipeline_v7_17_2.py

pip install pyserial opencv-python numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import sys
import os
import time
import datetime
import json
import re
import queue
import signal

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# THEME  — high-contrast dark palette
# ---------------------------------------------------------------------------
BG        = "#1a1a2e"   # deep navy background
BG2       = "#16213e"   # card / panel background
BG3       = "#0f3460"   # input / canvas background
ACCENT    = "#7c6af7"   # purple accent (buttons)
ACCENT2   = "#c4b5fd"   # light purple (values)
SUCCESS   = "#4ade80"   # bright green
WARNING   = "#fbbf24"   # amber
DANGER    = "#f87171"   # red
TEXT      = "#f0f0ff"   # near-white primary text
TEXT2     = "#c0c0d8"   # secondary text  (was #9090b0 — raised for contrast)
BORDER    = "#2a2a55"

# Button foregrounds — always near-black on bright BG for max contrast
BTN_LIGHT_FG = "#0a0a1a"

# Tab / menu high-contrast
TAB_BG_ACTIVE   = "#7c6af7"
TAB_FG_ACTIVE   = "#ffffff"
TAB_BG_INACTIVE = "#1e1e40"
TAB_FG_INACTIVE = "#c0c0d8"

FONT_FAMILY = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"

# ---------------------------------------------------------------------------
# SCRIPT NAMES (no dashes)
# ---------------------------------------------------------------------------
BRIDGE_SCRIPT   = "pavlovia_arduino_bridge_v5_2_2.py"
PIPELINE_SCRIPT = "psychophysiology_pipeline_v7_17_2.py"
WEBSOCKET_PORT  = 5678
ARDUINO_BAUDRATE = 115200
LOG_DIR         = "physiologging"

# ---------------------------------------------------------------------------
# CJMCU-6701 GSR SENSOR CONVERSION
# ---------------------------------------------------------------------------
# The CJMCU-6701 outputs a raw ADC voltage; the GUI converts it to µS for display.
# Formula: Rskin (Ω) = Rref * (Vcc - Vout) / Vout
#          Conductance (µS) = 1e6 / Rskin
# Tune these to match your hardware if readings look wrong.
GSR_VCC       = 5.0      # Supply voltage to sensor (3.3 or 5.0 V)
GSR_RREF      = 100_000  # Reference resistor in Ω — check your board (100kΩ typical for 5V systems)
GSR_ADC_MAX   = 32767    # ADS1115 16-bit raw counts (change to 4095 for 12-bit, 1023 for 10-bit)


def gsr_adc_to_uS(adc: float,
                  vcc: float = GSR_VCC,
                  rref: float = GSR_RREF,
                  adc_max: float = GSR_ADC_MAX) -> float:
    """Convert CJMCU-6701 raw ADC reading to skin conductance in µS.
    Formula: Rskin = Rref * (Vcc - Vout) / Vout;  conductance = 1e6 / Rskin
    """
    if adc <= 0 or adc >= adc_max:
        return 0.0
    vout = (adc / adc_max) * vcc
    r_skin = rref * (vcc - vout) / vout
    if r_skin <= 0:
        return 0.0
    return 1e6 / r_skin

# ---------------------------------------------------------------------------
# PAVLOVIA LATIN SQUARE
# ---------------------------------------------------------------------------
PAVLOVIA_BASE = "https://pavlovia.org/adamrcobb/"

# Experiment names per task per version
PAVLOVIA_TASKS = {
    "HA": {"A": "HA_A_final_v9", "B": "HA_B_final_v9",
            "C": "HA_C_final_v9", "D": "HA_D_final_v9"},
    "EX": {"A": "EX_A_final",    "B": "EX_B_final",
            "C": "EX_C_final",    "D": "EX_D_final"},
    "RR": {"A": "RR_A_final_v2", "B": "RR_B_final_v2",
            "C": "RR_C_final_v2", "D": "RR_D_final_v2"},
}
TASK_LABELS = {
    "HA": "Habituation / Acquisition",
    "EX": "Extinction",
    "RR": "Reinstatement / Retrieval",
}
TASK_ORDER  = ["HA", "EX", "RR"]   # canonical session order
VERSIONS    = ["A", "B", "C", "D"]
N_GROUPS    = 4


def ls_version(group: int, task_idx: int) -> str:
    """Cyclic latin square: group 1-4, task_idx 0=HA 1=EX 2=RR."""
    return VERSIONS[(group - 1 + task_idx) % N_GROUPS]


def ls_url(group: int, task_idx: int) -> str:
    task = TASK_ORDER[task_idx]
    ver  = ls_version(group, task_idx)
    return PAVLOVIA_BASE + PAVLOVIA_TASKS[task][ver].lower()


def group_from_subject(subject_id: str) -> int:
    """Derive group 1-4 from trailing digits of subject ID.
    Falls back to 1 if no numeric suffix found."""
    m = re.search(r'(\d+)\s*$', subject_id.strip())
    if m:
        n = int(m.group(1))
        return (n - 1) % N_GROUPS + 1
    return 1

WIZARD_STEPS_RA = [
    ("Subject ID",        "Enter participant ID and session notes."),
    ("Hardware Check",    "Verify Arduino, webcam, and electrode connections."),
    ("Signal Quality",    "Confirm clean ECG and GSR signals before starting."),
    ("Launch Experiment", "Start the bridge, then open Pavlovia in a browser."),
]

WIZARD_STEPS_PARTICIPANT = [
    ("Welcome",         "Let's get you set up. This takes about 5 minutes."),
    ("Electrode Guide", "Follow the pictures to attach the sensors."),
    ("Signal Check",    "We'll make sure everything is working."),
    ("Start Task",      "You're ready! The experiment will begin now."),
]

# ---------------------------------------------------------------------------
# ELECTRODE GUIDE TEXT
# ---------------------------------------------------------------------------
ELECTRODE_TEXT = """\
ELECTRODE PLACEMENT GUIDE
──────────────────────────────────────────────────────

  ECG (Heart) — 3 Sticky Electrode Pads
  ┌────────────────────────────────────┐
  │  RED lead    →  Right collarbone   │
  │  YELLOW lead →  Left lower ribcage │
  │  GREEN lead  →  Right lower abdomen│
  └────────────────────────────────────┘
  Tips:
  • Clean skin with an alcohol pad first. Let dry completely.
  • Press firmly for 10 seconds after applying each pad.
  • Avoid hairy areas — shave if needed.

  GSR (Sweat) — 2 Finger Clips / Velcro Bands
  ┌────────────────────────────────────┐
  │  Index finger  →  Left hand        │
  │  Middle finger →  Left hand        │
  └────────────────────────────────────┘
  Tips:
  • Attach to the middle section of each finger.
  • Snug but not tight — should not restrict blood flow.
  • Do not apply hand lotion before the session.

  Webcam
  ┌────────────────────────────────────┐
  │  Position so your face is centered │
  │  and well-lit from the front.      │
  │  Avoid strong backlighting.        │
  └────────────────────────────────────┘
"""


# ===========================================================================
# UTILITIES
# ===========================================================================

def styled_button(parent, text, command=None, style="primary", width=18, **kw):
    """Buttons: black text on light grey — always readable on macOS and Windows."""
    # Use a single high-contrast scheme that macOS Aqua cannot override:
    # near-black text on light grey background with a visible border.
    # Style only affects the left border accent color for visual variety.
    accent_colors = {
        "primary": "#5b4fe8",
        "success": "#16a34a",
        "danger":  "#dc2626",
        "ghost":   "#555577",
        "orange":  "#ea580c",
    }
    accent = accent_colors.get(style, accent_colors["primary"])
    btn = tk.Button(
        parent, text=text, command=command,
        bg="#d8d8d8", fg="#111111",
        activebackground="#c0c0c0", activeforeground="#111111",
        disabledforeground="#888888",
        relief="solid", bd=1,
        highlightbackground=accent, highlightcolor=accent,
        highlightthickness=2,
        font=(FONT_FAMILY, 11, "bold"),
        padx=14, pady=7, cursor="hand2", width=width,
        takefocus=0, **kw
    )
    return btn


def card_frame(parent, **kw):
    return tk.Frame(parent, bg=BG2, relief="flat", bd=0, **kw)


def label(parent, text, size=11, color=TEXT, bold=False, **kw):
    weight = "bold" if bold else "normal"
    bg = parent.cget("bg") if hasattr(parent, "cget") else BG
    return tk.Label(parent, text=text, bg=bg, fg=color,
                    font=(FONT_FAMILY, size, weight), **kw)


def separator(parent, color=BORDER):
    return tk.Frame(parent, bg=color, height=1)


# ===========================================================================
# SIGNAL CANVAS  — mini oscilloscope
# ===========================================================================

class SignalCanvas(tk.Canvas):
    HISTORY = 600

    def __init__(self, parent, channel="ECG", color=SUCCESS, **kw):
        kw.setdefault("bg", BG3)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", BORDER)
        super().__init__(parent, **kw)
        self.channel = channel
        self.color   = color
        self.data    = []
        self._pending_redraw = False
        self.bind("<Configure>", self._on_resize)

    def push(self, value):
        self.data.append(value)
        if len(self.data) > self.HISTORY:
            self.data = self.data[-self.HISTORY:]
        if not self._pending_redraw:
            self._pending_redraw = True
            self.after(33, self._redraw)   # throttle to ~30 fps

    def _on_resize(self, _=None):
        self.after(50, self._redraw)

    def _redraw(self, _=None):
        self._pending_redraw = False
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return
        # background grid lines
        for i in range(1, 4):
            y = int(h * i / 4)
            self.create_line(0, y, w, y, fill=BORDER, width=1)

        if len(self.data) < 2:
            self.create_text(w // 2, h // 2, text=f"Waiting for {self.channel}…",
                             fill=TEXT2, font=(FONT_FAMILY, 9))
            return

        pts = self.data[-w:] if len(self.data) > w else self.data
        mn, mx = min(pts), max(pts)
        span = mx - mn if mx != mn else 1.0
        pad = 6
        coords = []
        n = len(pts)
        for i, v in enumerate(pts):
            x = pad + (i / max(n - 1, 1)) * (w - 2 * pad)
            y = pad + (1.0 - (v - mn) / span) * (h - 2 * pad)
            coords += [x, y]
        self.create_line(coords, fill=self.color, width=2, smooth=True)
        # channel label
        self.create_text(8, 6, anchor="nw", text=self.channel,
                         fill=TEXT2, font=(FONT_FAMILY, 9, "bold"))
        # current value (only shown for GSR — ECG BPM shown in stat tile instead)
        if self.channel != "ECG":
            self.create_text(w - 6, 6, anchor="ne",
                             text=f"{pts[-1]:.1f}",
                             fill=self.color, font=(FONT_FAMILY, 9, "bold"))



def _estimate_bpm(ecg_buf, fs=250.0):
    """Estimate BPM from a DC-removed ECG buffer using scipy peak detection.
    Uses only the most recent 4 seconds to avoid startup noise.
    Returns None if insufficient data or no clear peaks found.
    ecg_buf: 1-D array of DC-removed ECG ADC counts.
    fs: sampling rate in Hz.
    """
    if not NUMPY_AVAILABLE or len(ecg_buf) < int(fs * 2):
        return None
    try:
        from scipy.signal import find_peaks, butter, sosfiltfilt
        # Use most recent 4s only — avoids startup noise contaminating estimate
        window = int(fs * 4)
        arr = np.asarray(ecg_buf[-window:], dtype=float)
        arr = arr - arr.mean()   # re-centre after slicing
        # Percentile clip for BPM: remove extreme spikes (beyond 0.5th–99.5th pct)
        # without shrinking QRS amplitude so peaks remain detectable.
        _lo, _hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        if _hi > _lo:
            arr = np.clip(arr, _lo, _hi)

        # Bandpass 5–20 Hz to isolate QRS
        nyq = fs / 2.0
        lo, hi = 5.0 / nyq, min(20.0 / nyq, 0.99)
        if lo >= hi:
            return None
        sos = butter(2, [lo, hi], btype='band', output='sos')
        filtered = sosfiltfilt(sos, arr)

        # Adaptive threshold: 80th percentile of |filtered| so it scales
        # with actual QRS amplitude rather than being misled by baseline shifts
        threshold = np.percentile(np.abs(filtered), 80)
        if threshold < 1e-6:
            return None

        # Min distance 450 ms prevents counting both flanks of a single QRS
        min_dist = max(1, int(fs * 0.45))
        peaks, props = find_peaks(filtered, distance=min_dist,
                                  height=threshold)
        if len(peaks) < 3:   # need at least 3 peaks for 2 reliable RR intervals
            return None

        rr_intervals = np.diff(peaks) / fs   # seconds
        rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
        if len(rr_valid) < 2:
            return None

        # Use median for robustness against occasional missed/extra beats
        bpm = 60.0 / float(np.median(rr_valid))
        if 30 <= bpm <= 200:
            return bpm
    except Exception:
        pass
    return None


# ===========================================================================
# ARDUINO READER  — background thread, feeds queues
# ===========================================================================

class ArduinoReader(threading.Thread):
    """Reads serial data from Arduino and fills ECG / GSR queues."""

    _TTL_RE = re.compile(r'^T\d+$')

    def __init__(self, port, baud=115200):
        super().__init__(daemon=True)
        self.port    = port
        self.baud    = baud
        self.ser     = None
        self.ecg_q        = queue.Queue(maxsize=1000)  # filtered ECG for display
        self.ecg_raw_q    = queue.Queue(maxsize=1000)  # raw ECG (unused currently)
        self.gsr_q        = queue.Queue(maxsize=1000)
        self.bpm_q        = queue.Queue(maxsize=100)
        self.running  = True
        self.connected = False
        self.error     = None
        self.live_bpm  = None   # most-recent BPM estimate (updated in poll)

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            time.sleep(2)
            self.connected = True
        except Exception as e:
            self.error = str(e)
            return

        while self.running:
            try:
                if self.ser.in_waiting:
                    raw = self.ser.readline()
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith(("#", "=", "//", ">")):
                        continue
                    if self._TTL_RE.match(line):
                        continue
                    parts = line.split(",")
                    # Arduino v3.3 serial format (5 fields):
                    #   0: ms           (timestamp)
                    #   1: rawGSR       (ADS1115 ADC counts)
                    #   2: GSR_uS       (already-converted µS)
                    #   3: rawECG       (Arduino A1 ADC counts)
                    #   4: ECG_mV       (converted mV)
                    #   5: eventMarker  (optional)
                    if len(parts) >= 5:
                        try:
                            gsr_us  = float(parts[2].strip())   # GSR_uS  ← col 2
                            ecg_raw = float(parts[3].strip())   # rawECG  ← col 3
                            if not self.ecg_q.full():
                                self.ecg_q.put_nowait(ecg_raw)
                            if not self.gsr_q.full():
                                self.gsr_q.put_nowait(gsr_us)
                        except ValueError:
                            pass
            except Exception:
                pass
            time.sleep(0.001)

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    def drain_ecg(self, max_n=100):
        out = []
        for _ in range(max_n):
            try:
                out.append(self.ecg_q.get_nowait())
            except queue.Empty:
                break
        return out

    def drain_gsr(self, max_n=100):
        out = []
        for _ in range(max_n):
            try:
                out.append(self.gsr_q.get_nowait())
            except queue.Empty:
                break
        return out


# ===========================================================================
# STATUS BAR
# ===========================================================================

class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#0d0d20", pady=5)
        self._items = {}

    def set(self, key, text, color=TEXT2):
        if key not in self._items:
            lbl = tk.Label(self, text="", bg="#0d0d20", fg=color,
                           font=(FONT_FAMILY, 9), padx=12)
            lbl.pack(side="left")
            self._items[key] = lbl
        self._items[key].config(text=text, fg=color)


# ===========================================================================
# SESSION MONITOR  — live dashboard
# ===========================================================================

class SessionMonitor(tk.Frame):
    def __init__(self, parent, app_ref=None):
        super().__init__(parent, bg=BG)
        self._app = app_ref
        self._elapsed  = 0
        self._running  = False
        self._timer_id = None
        self._ttl_count = 0
        self._gsr_ema   = 0.0   # EMA state for GSR display LP filter
        self._ecg_buf   = []    # rolling ECG buffer for BPM estimation
        self._ecg_sos   = None  # Butterworth bandpass SOS (built on first use)
        self._ecg_zi    = None  # persistent filter state across poll calls
        self._build()

    def _build(self):
        # ── Stat tiles ────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=14, pady=(14, 6))

        self._stats = {}
        tiles = [
            ("state",   "State",     "IDLE",     TEXT2),
            ("elapsed", "Elapsed",   "00:00",    TEXT),
            ("fps",     "Video FPS", "—",        TEXT),
            ("frames",  "Frames",    "0",        TEXT),
            ("ecg_bpm", "ECG HR",    "—",        SUCCESS),
            ("gsr",     "GSR µS",    "—",        ACCENT2),
            ("markers", "Markers",   "0",        WARNING),
            ("ttl",     "TTL",       "0",        ACCENT2),
        ]
        self._gsr_range_lbl = None
        for col, (key, title, init, color) in enumerate(tiles):
            f = tk.Frame(top, bg=BG2, padx=10, pady=8)
            f.grid(row=0, column=col, padx=3, pady=2, sticky="nsew")
            top.columnconfigure(col, weight=1)
            tk.Label(f, text=title, bg=BG2, fg=TEXT2,
                     font=(FONT_FAMILY, 8, "bold")).pack()
            val = tk.Label(f, text=init, bg=BG2, fg=color,
                           font=(FONT_FAMILY, 15, "bold"))
            val.pack()
            self._stats[key] = val
            # GSR tile gets an extra range-status badge
            if key == "gsr":
                self._gsr_range_lbl = tk.Label(
                    f, text="", bg=BG2, fg=TEXT2,
                    font=(FONT_FAMILY, 8, "bold"))
                self._gsr_range_lbl.pack()

        # ── Live signal canvases ──────────────────────────────────────────
        self._sig_outer = tk.Frame(self, bg=BG2, padx=8, pady=8)
        self._sig_outer.pack(fill="x", padx=14, pady=4)
        sig_outer = self._sig_outer
        tk.Label(sig_outer, text="LIVE SIGNALS", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")

        self.ecg_canvas = SignalCanvas(sig_outer, channel="ECG", color=SUCCESS, height=90)
        self.ecg_canvas.pack(fill="x", pady=(4, 2))
        self.gsr_canvas = SignalCanvas(sig_outer, channel="GSR", color=ACCENT2, height=70)
        self.gsr_canvas.pack(fill="x", pady=(2, 4))

        # ── Bridge-active notice (shown instead of canvases while bridge runs)
        self._bridge_notice = tk.Frame(self, bg=BG2, padx=16, pady=14)
        # (not packed by default — shown by set_canvas_mode(False))
        tk.Label(self._bridge_notice,
                 text="● Bridge Running — Signals Recording to CSV",
                 bg=BG2, fg=SUCCESS,
                 font=(FONT_FAMILY, 13, "bold")).pack()
        tk.Label(self._bridge_notice,
                 text="Live waveforms are available after stopping the bridge.\n"
                      "ECG HR and GSR µS are updated every 2 seconds from the CSV.",
                 bg=BG2, fg=TEXT2,
                 font=(FONT_FAMILY, 10)).pack(pady=(6, 0))

        # ── GSR range legend ──────────────────────────────────────────────
        legend = tk.Frame(self, bg=BG, padx=8)
        legend.pack(fill="x", padx=14, pady=(2, 0))
        tk.Label(legend, text="GSR range:", bg=BG, fg=TEXT2,
                 font=(FONT_FAMILY, 8)).pack(side="left", padx=(0, 6))
        for txt, col in [
            ("< 0.5 µS  No signal",   DANGER),
            ("0.5–3 µS  Low",         WARNING),
            ("3–38 µS  Normal ✓",     SUCCESS),
            ("38–100 µS  High",       WARNING),
            ("> 100 µS  Artefact?",   DANGER),
        ]:
            tk.Label(legend, text=txt, bg=BG, fg=col,
                     font=(FONT_FAMILY, 8, "bold")).pack(side="left", padx=8)

        # ── Port selector for live monitoring ─────────────────────────────
        port_frame = tk.Frame(self, bg=BG2, padx=8, pady=6)
        port_frame.pack(fill="x", padx=14, pady=2)
        tk.Label(port_frame, text="Arduino Port for live monitoring:",
                 bg=BG2, fg=TEXT2, font=(FONT_FAMILY, 9)).pack(side="left")
        self._port_var = tk.StringVar()
        self._port_cb  = ttk.Combobox(port_frame, textvariable=self._port_var,
                                       state="readonly", width=22,
                                       font=(FONT_FAMILY, 10))
        self._port_cb.pack(side="left", padx=6)
        styled_button(port_frame, "↻", self._refresh_ports,
                      style="ghost", width=3).pack(side="left")
        self._mon_btn = styled_button(port_frame, "▶ Start Monitoring",
                                       self._toggle_monitoring,
                                       style="success", width=18)
        self._mon_btn.pack(side="left", padx=6)
        self._mon_status = tk.Label(port_frame, text="", bg=BG2, fg=TEXT2,
                                     font=(FONT_FAMILY, 9))
        self._mon_status.pack(side="left", padx=6)
        self._reader = None
        self.after(150, self._refresh_ports)

        # ── Event marker log ──────────────────────────────────────────────
        log_outer = tk.Frame(self, bg=BG2, padx=8, pady=6)
        log_outer.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        tk.Label(log_outer, text="EVENT MARKERS", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")
        self._log = tk.Text(log_outer, bg=BG3, fg=TEXT,
                             font=("Courier", 10), relief="flat", bd=0,
                             state="disabled", height=7,
                             insertbackground=TEXT, selectbackground=ACCENT)
        self._log.pack(fill="both", expand=True, pady=(4, 0))

    # ── Port / monitoring controls ─────────────────────────────────────────

    def _refresh_ports(self):
        if not SERIAL_AVAILABLE:
            self._port_cb["values"] = ["pyserial not installed"]
            return
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self._port_cb["values"] = ports or ["No ports found"]
        if ports:
            self._port_var.set(ports[0])

    def _toggle_monitoring(self):
        if self._reader and self._reader.running:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self):
        if getattr(self, '_bridge_active', False):
            self._mon_status.config(
                text="Bridge holds serial port — stop bridge first", fg=WARNING)
            return
        port = self._port_var.get()
        if not port or port in ("No ports found", "pyserial not installed"):
            self._mon_status.config(text="No valid port selected", fg=DANGER)
            return
        if not SERIAL_AVAILABLE:
            self._mon_status.config(text="pyserial not installed", fg=DANGER)
            return
        if self._reader:
            self._reader.stop()
        self._reader = ArduinoReader(port, ARDUINO_BAUDRATE)
        self._reader.start()
        self._mon_btn.config(text="⏹ Stop Monitoring", bg="#d8d8d8", fg="#111111")
        self._mon_status.config(text=f"Connecting to {port}…", fg=WARNING)
        self._poll_reader()

    def _stop_monitoring(self):
        if self._reader:
            self._reader.stop()
            self._reader = None
        self._mon_btn.config(text="▶ Start Monitoring", bg="#d8d8d8", fg="#111111")
        self._mon_status.config(text="Monitoring stopped", fg=TEXT2)
        # Reset filter state so next session starts clean
        self._ecg_sos = None
        self._ecg_zi  = None
        self._ecg_buf = []
        self._gsr_ema = 0.0  # reset so tile shows clean state on next connect

    def _poll_reader(self):
        """Called every 50 ms to drain ArduinoReader queues → canvases."""
        if not self._reader:
            return
        if not self._reader.running:
            return

        # Update connection status on first successful read
        if self._reader.connected and "Connecting" in self._mon_status.cget("text"):
            self._mon_status.config(
                text=f"● Live: {self._reader.port}", fg=SUCCESS)

        if self._reader.error:
            self._mon_status.config(
                text=f"Error: {self._reader.error}", fg=DANGER)
            self._stop_monitoring()
            return

        ecg_vals = self._reader.drain_ecg(80)
        gsr_vals = self._reader.drain_gsr(80)

        # ── ECG: continuous bandpass filter (0.5–40 Hz) for medical-grade display
        if ecg_vals and NUMPY_AVAILABLE:
            try:
                from scipy.signal import butter, sosfilt, sosfilt_zi
                _scipy_ok2 = True
            except ImportError:
                _scipy_ok2 = False
            if not _scipy_ok2:
                for v in ecg_vals:
                    self.ecg_canvas.push(float(v))
            # Build filter once
            if _scipy_ok2 and self._ecg_sos is None:
                try:
                    nyq = 125.0  # 250 Hz / 2
                    self._ecg_sos = butter(2, [0.5/nyq, 40.0/nyq],
                                           btype='band', output='sos')
                    self._ecg_zi = sosfilt_zi(self._ecg_sos) * 0.0  # start from zero
                except Exception:
                    self._ecg_sos = None

            if self._ecg_sos is not None:
                chunk = np.array(ecg_vals, dtype=float)
                # DC subtract before filter — eliminates startup transient
                chunk = chunk - chunk.mean()
                # Warm zi to first sample so no step-input ringing
                if np.all(self._ecg_zi == 0) and len(chunk) > 0:
                    self._ecg_zi = sosfilt_zi(self._ecg_sos) * chunk[0]
                filtered, self._ecg_zi = sosfilt(
                    self._ecg_sos, chunk, zi=self._ecg_zi)
                for v in filtered:
                    self.ecg_canvas.push(float(v))
                # Accumulate DC-removed (unfiltered) for BPM — needs full buffer
                self._ecg_buf.extend(ecg_vals)
                if len(self._ecg_buf) > 1000:
                    self._ecg_buf = self._ecg_buf[-1000:]
                buf_dc = np.array(self._ecg_buf, dtype=float)
                buf_dc -= buf_dc.mean()
                bpm = _estimate_bpm(buf_dc, fs=250.0)
                if bpm is not None:
                    self.update_stat("ecg_bpm", f"{bpm:.0f}")
                elif len(self._ecg_buf) < 500:
                    self.update_stat("ecg_bpm", "…", WARNING)
            else:
                # scipy unavailable — fall back to DC removal only
                self._ecg_buf.extend(ecg_vals)
                if len(self._ecg_buf) > 1000:
                    self._ecg_buf = self._ecg_buf[-1000:]
                buf = np.array(self._ecg_buf, dtype=float) - np.mean(self._ecg_buf)
                n_new = min(len(ecg_vals), len(buf))
                for v in buf[-n_new:]:
                    self.ecg_canvas.push(float(v))

        # ── GSR: ArduinoReader queues GSR_uS (parts[2]) — already in µS, no conversion needed
        for v in gsr_vals:
            if 0.05 <= v <= 200.0:  # physiological plausibility gate
                if self._gsr_ema == 0.0:
                    self._gsr_ema = v  # warm-start on first valid reading
                else:
                    self._gsr_ema = self._gsr_ema * 0.85 + v * 0.15
                self.gsr_canvas.push(self._gsr_ema)

        # Update live GSR tile with range badge
        if gsr_vals:
            self.update_gsr(self._gsr_ema)

        self.after(50, self._poll_reader)

    # ── Timer ──────────────────────────────────────────────────────────────

    def start_timer(self):
        self._elapsed = 0
        self._running = True
        self._csv_bpm_path = None   # set by HPMApp when bridge starts
        self._csv_file_pos  = 0     # byte offset for incremental CSV reading
        self._csv_header    = None  # cached header row
        self._csv_ecg_buf   = []    # rolling buffer for BPM (6s)
        self._csv_gsr_last  = 0.0   # last valid GSR reading
        # Pre-build the ECG bandpass filter so first CSV poll can use it
        if NUMPY_AVAILABLE and self._ecg_sos is None:
            try:
                from scipy.signal import butter, sosfilt_zi
                nyq = 125.0
                self._ecg_sos = butter(2, [0.5/nyq, 40.0/nyq],
                                       btype='band', output='sos')
                self._ecg_zi = sosfilt_zi(self._ecg_sos) * 0.0
            except Exception:
                self._ecg_sos = None
        self._tick()
        self._poll_bridge_bpm()

    def stop_timer(self):
        self._running = False
        if self._timer_id:
            self.after_cancel(self._timer_id)

    def set_csv_path(self, path):
        """Called by HPMApp when a new physio CSV is created."""
        self._csv_bpm_path  = path
        self._csv_file_pos  = 0    # reset seek position for new file
        self._csv_header    = None
        self._csv_ecg_buf   = []
        self._ecg_buf       = []
        self._ecg_sos       = None
        self._ecg_zi        = None
        self._gsr_ema       = 0.0

    def _poll_bridge_bpm(self):
        """Every 250ms: feed ECG+GSR canvases and BPM from live CSV.
        Reads the last 65 rows (250ms of data at 250Hz) each call.
        This is the primary canvas source when the bridge is running.
        """
        if not self._running:
            return
        path = getattr(self, '_csv_bpm_path', None)
        # Fallback: scan LOG_DIR for most recently modified physio CSV
        if not path or not os.path.exists(path):
            import glob as _glob
            # Search relative to bridge script directory AND hpm_gui.py directory
            _here = os.path.dirname(os.path.abspath(__file__))
            _search_dirs = [_here]
            if self._app and self._app._bridge_mgr:
                _bd = os.path.dirname(
                    os.path.abspath(self._app._bridge_mgr.script_path))
                if _bd not in _search_dirs:
                    _search_dirs.append(_bd)
            _candidates = []
            for _d in _search_dirs:
                _candidates.extend(
                    _glob.glob(os.path.join(_d, LOG_DIR, '*_physiodata.csv')))
            if _candidates:
                path = max(_candidates, key=os.path.getmtime)
                self._csv_bpm_path = path
        if path and os.path.exists(path) and NUMPY_AVAILABLE:
            try:
                import csv as _csv
                new_rows = []
                with open(path, 'r', newline='', errors='ignore') as f:
                    # Read header once, then seek to last known position
                    if self._csv_header is None:
                        self._csv_header = next(_csv.reader(f), None)
                        self._csv_file_pos = f.tell()
                    else:
                        f.seek(0, 2)  # end of file
                        eof = f.tell()
                        if eof <= self._csv_file_pos:
                            self.after(2000, self._poll_bridge_bpm)
                            return
                        f.seek(self._csv_file_pos)
                        for row in _csv.reader(f):
                            new_rows.append(row)
                        self._csv_file_pos = f.tell()
                header = self._csv_header
                rows   = new_rows
                if not header or not rows:
                    self.after(2000, self._poll_bridge_bpm)
                    return

                # ── ECG canvas + BPM ─────────────────────────────────────
                ecg_col = next((i for i, h in enumerate(header)
                               if h in ('RawECG', 'ECG_mV', 'ECGmV')), None)
                if ecg_col is not None:
                    # Canvas: push all new rows received this poll
                    new_ecg = []
                    for r in rows:
                        try:
                            new_ecg.append(float(r[ecg_col]))
                        except (ValueError, IndexError):
                            pass
                    # No canvas push during bridge — canvases are hidden
                    pass

                    # BPM: accumulate into rolling 6s buffer
                    self._csv_ecg_buf.extend(new_ecg)
                    if len(self._csv_ecg_buf) > 1500:
                        self._csv_ecg_buf = self._csv_ecg_buf[-1500:]
                    if len(self._csv_ecg_buf) > 200:
                        arr = np.array(self._csv_ecg_buf, dtype=float)
                        arr -= arr.mean()
                        _lo, _hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
                        if _hi > _lo:
                            arr = np.clip(arr, _lo, _hi)
                        bpm = _estimate_bpm(arr, fs=250.0)
                        if bpm is not None:
                            self.update_stat('ecg_bpm', f'{bpm:.0f}')

                # ── GSR canvas + tile ─────────────────────────────────────
                gsr_col = next((i for i, h in enumerate(header)
                               if h in ('GSR_uS', 'GSRuS', 'RawGSR')), None)
                if gsr_col is not None:
                    for r in rows:
                        try:
                            v_uS = float(r[gsr_col])
                            # CSV column GSR_uS is already in µS — use directly
                            if 0.05 <= v_uS <= 200.0:
                                if self._gsr_ema == 0.0:
                                    self._gsr_ema = v_uS
                                else:
                                    self._gsr_ema = self._gsr_ema * 0.85 + v_uS * 0.15
                        except (ValueError, IndexError):
                            pass
                    self.update_gsr(self._gsr_ema)  # tile + badge only

            except Exception:
                pass
        self.after(2000, self._poll_bridge_bpm)

    def _tick(self):
        if not self._running:
            return
        self._elapsed += 1
        m, s = divmod(self._elapsed, 60)
        self._stats["elapsed"].config(text=f"{m:02d}:{s:02d}")
        self._timer_id = self.after(1000, self._tick)

    # ── Public update API ──────────────────────────────────────────────────

    # GSR physiological range (µS) for colour coding
    GSR_RANGE_LOW  = 0.5    # below = disconnected / very dry
    GSR_RANGE_OK   = 3.0    # lower bound of expected resting range
    GSR_RANGE_HIGH = 38.0   # upper bound of expected resting range
    GSR_RANGE_MAX  = 100.0  # above = possible artefact

    def update_stat(self, key, text, color=None):
        if key in self._stats:
            cfg = {"text": text}
            if color:
                cfg["fg"] = color
            self._stats[key].config(**cfg)

    def set_canvas_mode(self, visible: bool):
        """Show or hide live signal canvases.
        Hidden during bridge (data recorded to CSV).
        Shown when ArduinoReader is active.
        """
        if visible:
            self._sig_outer.pack(fill="x", padx=14, pady=4)
            self._bridge_notice.pack_forget()
        else:
            self._sig_outer.pack_forget()
            self._bridge_notice.pack(fill="x", padx=14, pady=8)

    def update_gsr(self, us_value: float):
        """Update GSR tile with µS value and colour-coded range badge."""
        self._stats["gsr"].config(text=f"{us_value:.2f}")
        if self._gsr_range_lbl is None:
            return
        if us_value < self.GSR_RANGE_LOW:
            badge, color = "● NO SIGNAL", DANGER
        elif us_value < self.GSR_RANGE_OK:
            badge, color = "▼ LOW", WARNING
        elif us_value <= self.GSR_RANGE_HIGH:
            badge, color = "✓ NORMAL", SUCCESS
        elif us_value <= self.GSR_RANGE_MAX:
            badge, color = "▲ HIGH", WARNING
        else:
            badge, color = "!! ARTIFACT?", DANGER
        self._gsr_range_lbl.config(text=badge, fg=color)

    def log_marker(self, ts, marker):
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}]  {marker}\n")
        self._log.see("end")
        self._log.config(state="disabled")
        try:
            count = int(self._stats["markers"].cget("text"))
        except ValueError:
            count = 0
        self._stats["markers"].config(text=str(count + 1))

    def increment_ttl(self):
        self._ttl_count += 1
        self._stats["ttl"].config(text=str(self._ttl_count))

    def push_ecg(self, v):
        self.ecg_canvas.push(v)

    def push_gsr(self, v):
        self.gsr_canvas.push(v)


# ===========================================================================
# SIGNAL QUALITY PAGE  (popup used in wizard)
# ===========================================================================

class SignalQualityPage(tk.Frame):
    def __init__(self, parent, on_pass=None):
        super().__init__(parent, bg=BG)
        self.on_pass = on_pass
        self.reader  = None
        self._ecg_ok = False
        self._gsr_ok = False
        self._build()

    def _build(self):
        tk.Label(self, text="Signal Quality Check", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 16, "bold")).pack(pady=(18, 2))
        tk.Label(self, text="Confirm clean ECG and GSR signals before proceeding.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack()

        # Port row
        sel = card_frame(self)
        sel.pack(fill="x", padx=20, pady=10)
        port_row = tk.Frame(sel, bg=BG2)
        port_row.pack(fill="x", padx=10, pady=8)
        tk.Label(port_row, text="Port:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self._port_var = tk.StringVar()
        self._port_cb  = ttk.Combobox(port_row, textvariable=self._port_var,
                                       state="readonly", width=24,
                                       font=(FONT_FAMILY, 10))
        self._port_cb.pack(side="left", padx=6)
        styled_button(port_row, "↻ Refresh", self._refresh,
                      style="ghost", width=10).pack(side="left", padx=4)
        styled_button(port_row, "▶ Connect",  self._connect,
                      style="primary", width=10).pack(side="left")
        self._conn_lbl = tk.Label(port_row, text="", bg=BG2, fg=TEXT2,
                                   font=(FONT_FAMILY, 9))
        self._conn_lbl.pack(side="left", padx=8)
        self.after(150, self._refresh)  # delay until Toplevel is mapped

        # Canvases
        cv_frame = card_frame(self)
        cv_frame.pack(fill="x", padx=20, pady=4)
        tk.Label(cv_frame, text="LIVE PREVIEW", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.ecg_cv = SignalCanvas(cv_frame, channel="ECG", color=SUCCESS, height=90)
        self.ecg_cv.pack(fill="x", padx=8, pady=2)
        self.gsr_cv = SignalCanvas(cv_frame, channel="GSR", color=ACCENT2, height=70)
        self.gsr_cv.pack(fill="x", padx=8, pady=(2, 8))

        # Indicators
        qi = card_frame(self)
        qi.pack(fill="x", padx=20, pady=4)
        qi_row = tk.Frame(qi, bg=BG2)
        qi_row.pack(padx=10, pady=10)
        self._ecg_ind = self._mk_indicator(qi_row, "ECG")
        self._ecg_ind.pack(side="left", padx=20)
        self._gsr_ind = self._mk_indicator(qi_row, "GSR")
        self._gsr_ind.pack(side="left", padx=20)

        # Continue
        self._cont_btn = styled_button(self, "Continue  →", self._continue,
                                        style="success", width=20)
        self._cont_btn.pack(pady=14)
        self._cont_btn.config(state="disabled")

    def _mk_indicator(self, parent, ch):
        f = tk.Frame(parent, bg=BG2)
        dot = tk.Label(f, text="●", fg=TEXT2, bg=BG2,
                       font=(FONT_FAMILY, 20))
        dot.pack()
        tk.Label(f, text=ch, fg=TEXT2, bg=BG2,
                 font=(FONT_FAMILY, 10, "bold")).pack()
        tk.Label(f, text="Waiting…", fg=TEXT2, bg=BG2,
                 font=(FONT_FAMILY, 8)).pack()
        f._dot   = dot
        f._ok    = False
        return f

    def _refresh(self):
        if not SERIAL_AVAILABLE:
            self._port_cb["values"] = ["pyserial not installed"]
            return
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self._port_cb["values"] = ports or ["No ports found"]
        if ports:
            self._port_var.set(ports[0])

    def _connect(self):
        port = self._port_var.get()
        if not port or "No ports" in port or "not installed" in port:
            messagebox.showerror("Port Error", "Select a valid Arduino port first.")
            return
        if self.reader:
            self.reader.stop()
        # Reset all state so a fresh connection starts clean
        self._gsr_ema  = 0.0
        self._ecg_buf  = []
        self._ecg_sos  = None
        self._ecg_zi   = None
        self.gsr_cv.data.clear()
        self.ecg_cv.data.clear()
        self.reader = ArduinoReader(port, ARDUINO_BAUDRATE)
        self.reader.start()
        self._conn_lbl.config(text=f"Connecting to {port}…", fg=WARNING)
        self._poll()

    def _poll(self):
        if not self.reader or not self.reader.running:
            return

        if self.reader.error:
            self._conn_lbl.config(text=f"Error: {self.reader.error}", fg=DANGER)
            return

        if self.reader.connected and "Connecting" in self._conn_lbl.cget("text"):
            self._conn_lbl.config(text=f"● {self.reader.port}", fg=SUCCESS)

        ecg = self.reader.drain_ecg(60)
        gsr = self.reader.drain_gsr(60)

        # ── ECG: continuous 0.5–40 Hz bandpass for medical-grade display ──
        self._ecg_buf = getattr(self, '_ecg_buf', [])
        self._ecg_buf.extend(ecg)
        if len(self._ecg_buf) > 1000:
            self._ecg_buf = self._ecg_buf[-1000:]

        if ecg and NUMPY_AVAILABLE:
            try:
                from scipy.signal import butter, sosfilt, sosfilt_zi
                _scipy_ok = True
            except ImportError:
                _scipy_ok = False
            if _scipy_ok:
                if not hasattr(self, '_ecg_sos') or self._ecg_sos is None:
                    try:
                        nyq = 125.0
                        self._ecg_sos = butter(2, [0.5/nyq, 40.0/nyq],
                                               btype='band', output='sos')
                        self._ecg_zi = sosfilt_zi(self._ecg_sos) * 0.0
                    except Exception:
                        self._ecg_sos = None
                if getattr(self, '_ecg_sos', None) is not None:
                    chunk = np.array(ecg, dtype=float)
                    chunk = chunk - chunk.mean()  # DC removal before filter
                    _zi = getattr(self, '_ecg_zi', None)
                    if _zi is None or (np.all(_zi == 0) and len(chunk) > 0):
                        self._ecg_zi = sosfilt_zi(self._ecg_sos) * chunk[0]
                    filtered, self._ecg_zi = sosfilt(
                        self._ecg_sos, chunk, zi=self._ecg_zi)
                    for v in filtered:
                        self.ecg_cv.push(float(v))
                else:
                    buf = np.array(self._ecg_buf, dtype=float)
                    buf -= buf.mean()
                    n_new = min(len(ecg), len(buf))
                    for v in buf[-n_new:]:
                        self.ecg_cv.push(float(v))
            else:
                # scipy not available — push raw values DC-removed
                buf = np.array(self._ecg_buf, dtype=float)
                buf -= buf.mean()
                n_new = min(len(ecg), len(buf))
                for v in buf[-n_new:]:
                    self.ecg_cv.push(float(v))

        # ── GSR: EMA with warm-start + corrupt line rejection ─────────────
        for v in gsr:
            if 0.05 <= v <= 200.0:
                _ema = getattr(self, '_gsr_ema', None)
                if _ema is None or _ema == 0.0:
                    self._gsr_ema = v   # warm-start: jump to first valid reading
                else:
                    self._gsr_ema = _ema * 0.85 + v * 0.15
            self.gsr_cv.push(getattr(self, '_gsr_ema', 0.0))

        # ── Quality heuristics ────────────────────────────────────────────
        # ECG quality: sketch outputs RawECG (ADC counts or mV depending on version)
        # Use a relative threshold: std > 1% of the signal range (peak-to-peak)
        if NUMPY_AVAILABLE and len(self._ecg_buf) >= 50:
            buf = np.array(self._ecg_buf, dtype=float)
            buf = buf - buf.mean()
            ecg_std  = float(np.std(buf))
            ecg_range = float(buf.max() - buf.min())
            # Signal present if std > 1% of range AND range is non-trivial
            signal_ok = ecg_range > 0 and ecg_std > 0.01 * ecg_range and ecg_std > 0.05
            bpm = _estimate_bpm(buf, fs=250.0)
            if signal_ok and bpm is not None:
                self._ecg_ind._dot.config(fg=SUCCESS)
                self._ecg_ind._ok = True
                self._ecg_ind.winfo_children()[2].config(
                    text=f"{bpm:.0f} BPM", fg=SUCCESS)
            elif signal_ok:
                self._ecg_ind._dot.config(fg=WARNING)
                self._ecg_ind.winfo_children()[2].config(
                    text="Signal OK — detecting…", fg=WARNING)
                self._ecg_ind._ok = False
            else:
                self._ecg_ind._dot.config(fg=DANGER)
                self._ecg_ind.winfo_children()[2].config(
                    text="Weak signal", fg=DANGER)
                self._ecg_ind._ok = False

        # GSR values from sketch are already in µS; valid range 0.1–60 µS
        if len(gsr) >= 5:
            gsr_mean = float(np.mean(gsr)) if NUMPY_AVAILABLE else sum(gsr) / len(gsr)
            if gsr_mean > 0.1:
                self._gsr_ind._dot.config(fg=SUCCESS)
                self._gsr_ind._ok = True
                self._gsr_ind.winfo_children()[2].config(
                    text=f"OK  ({gsr_mean:.2f} µS)", fg=SUCCESS)
            else:
                self._gsr_ind._dot.config(fg=WARNING)
                self._gsr_ind.winfo_children()[2].config(
                    text="Check contacts", fg=WARNING)

        both = self._ecg_ind._ok and self._gsr_ind._ok
        self._cont_btn.config(state="normal" if both else "disabled")
        self.after(80, self._poll)

    def _continue(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        if self.on_pass:
            self.on_pass()


# ===========================================================================
# SETUP WIZARD
# ===========================================================================

class SetupWizard(tk.Frame):
    def __init__(self, parent, steps, mode="ra", on_complete=None, subject_var=None):
        super().__init__(parent, bg=BG)
        self.steps       = steps
        self.mode        = mode
        self.on_complete = on_complete
        self.subject_var = subject_var
        self._step       = 0
        self._cal_elapsed = 0
        self._build()
        self._show_step(0)

    # ── Scaffold ─────────────────────────────────────────────────────────

    def _build(self):
        # Progress dots
        self._strip = tk.Frame(self, bg=BG, pady=10)
        self._strip.pack(fill="x", padx=20)
        self._dots = []
        for i, (title, _) in enumerate(self.steps):
            col = tk.Frame(self._strip, bg=BG)
            col.pack(side="left", expand=True)
            dot = tk.Label(col, text="●", fg=BORDER, bg=BG, font=(FONT_FAMILY, 16))
            dot.pack()
            tk.Label(col, text=title, fg=TEXT2, bg=BG,
                     font=(FONT_FAMILY, 8), wraplength=80, justify="center").pack()
            self._dots.append(dot)

        separator(self).pack(fill="x", padx=20)

        # Content
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True, padx=20, pady=10)

        # Nav buttons
        nav = tk.Frame(self, bg=BG, pady=10)
        nav.pack(fill="x", padx=20)
        self._back_btn = styled_button(nav, "← Back", self._prev, style="ghost", width=12)
        self._back_btn.pack(side="left")
        self._next_btn = styled_button(nav, "Next →", self._next, style="primary", width=14)
        self._next_btn.pack(side="right")

    def _clear_content(self):
        for w in self._content.winfo_children():
            w.destroy()

    # ── Step router ───────────────────────────────────────────────────────

    def _show_step(self, idx):
        self._clear_content()
        title, desc = self.steps[idx]

        for i, dot in enumerate(self._dots):
            dot.config(fg=SUCCESS if i < idx else (ACCENT if i == idx else BORDER))

        self._back_btn.config(state="normal" if idx > 0 else "disabled")
        is_last = idx == len(self.steps) - 1
        self._next_btn.config(text="Launch ▶" if is_last else "Next →",
                               state="normal")

        tk.Label(self._content, text=title, bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 20, "bold")).pack(anchor="w", pady=(6, 2))
        tk.Label(self._content, text=desc, bg=BG, fg=TEXT2,
                 font=(FONT_FAMILY, 10)).pack(anchor="w", pady=(0, 10))
        separator(self._content).pack(fill="x", pady=6)

        t = title.lower()
        if "subject" in t or "welcome" in t:
            self._step_subject()
        elif "hardware" in t:
            self._step_hardware()
        elif "electrode" in t:
            self._step_electrode()
        elif "signal" in t:
            self._step_signal()
        elif "calibrat" in t:
            self._step_calibration()
        elif "launch" in t or "start" in t:
            self._step_launch()
        else:
            tk.Label(self._content, text="Ready to proceed.",
                     bg=BG, fg=TEXT2, font=(FONT_FAMILY, 11)).pack(pady=20)

    # ── Step builders ─────────────────────────────────────────────────────

    def _step_subject(self):
        if self.mode == "participant":
            tk.Label(self._content,
                     text="Welcome to the HPM Study!\n\n"
                          "This session will record your heart rate and skin\n"
                          "conductance while you complete a short computer task.\n\n"
                          "Enter your Participant ID below (given by your researcher).",
                     bg=BG, fg=TEXT, font=(FONT_FAMILY, 11),
                     justify="left").pack(anchor="w")
        else:
            tk.Label(self._content, text="Session Setup", bg=BG, fg=TEXT,
                     font=(FONT_FAMILY, 13, "bold")).pack(anchor="w")

        frm = card_frame(self._content)
        frm.pack(fill="x", pady=10)

        row = tk.Frame(frm, bg=BG2)
        row.pack(fill="x", padx=12, pady=10)
        tk.Label(row, text="Participant ID:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold"), width=16, anchor="w").pack(side="left")
        ent = tk.Entry(row, textvariable=self.subject_var,
                       font=(FONT_FAMILY, 12), bg=BG3, fg=TEXT,
                       insertbackground=TEXT, relief="flat", bd=4, width=20)
        ent.pack(side="left", padx=8)
        ent.focus_set()

        if self.mode == "ra":
            row2 = tk.Frame(frm, bg=BG2)
            row2.pack(fill="x", padx=12, pady=(0, 4))
            tk.Label(row2, text="Session Notes:", bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 11, "bold"), width=16, anchor="w").pack(side="left", anchor="n")
            self._notes = tk.Text(frm, height=3, bg=BG3, fg=TEXT, font=(FONT_FAMILY, 11),
                                   relief="flat", bd=4, insertbackground=TEXT)
            self._notes.pack(fill="x", padx=12, pady=(0, 10))

    def _step_hardware(self):
        items = [
            ("Arduino",       "USB cable plugged in — green power LED on."),
            ("Webcam",        "USB webcam connected and positioned at face level."),
            ("Electrodes",    "Electrode leads attached to the sensor board."),
            ("Faraday Cage",  "Lid secured on enclosure (copper-lined box)."),
        ]
        frm = card_frame(self._content)
        frm.pack(fill="x", pady=8)
        self._hw_vars = {}

        for key, desc in items:
            row = tk.Frame(frm, bg=BG2)
            row.pack(fill="x", padx=12, pady=6)
            var = tk.BooleanVar()
            cb = tk.Checkbutton(
                row, variable=var, bg=BG2,
                activebackground=BG2, selectcolor="#3a3a80",
                fg=TEXT, font=(FONT_FAMILY, 11, "bold"),
                text=f"  {key}", anchor="w",
                disabledforeground=TEXT2,
                highlightthickness=0, takefocus=0,
            )
            cb.pack(side="left")
            tk.Label(row, text=f"— {desc}", bg=BG2, fg=TEXT2,
                     font=(FONT_FAMILY, 10)).pack(side="left", padx=6)
            self._hw_vars[key] = var

        self._next_btn.config(state="disabled")

        def _check(*_):
            ok = all(v.get() for v in self._hw_vars.values())
            self._next_btn.config(state="normal" if ok else "disabled")

        for v in self._hw_vars.values():
            v.trace_add("write", _check)

    def _step_electrode(self):
        txt = tk.Text(self._content, bg=BG3, fg=TEXT, font=("Courier", 10),
                      relief="flat", bd=0, height=22, wrap="word",
                      insertbackground=TEXT)
        txt.insert("1.0", ELECTRODE_TEXT)
        txt.config(state="disabled")
        txt.pack(fill="x", pady=4)

    def _step_signal(self):
        tk.Label(self._content,
                 text="Click the button below to open the live signal monitor.\n"
                      "Confirm both ECG and GSR show clean traces, then close it to continue.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack(anchor="w")
        self._next_btn.config(state="disabled")

        def open_sig_win():
            win = tk.Toplevel(self._content)
            win.title("Signal Quality Check")
            win.configure(bg=BG)
            win.geometry("660x540")

            def on_pass():
                win.destroy()
                self._next_btn.config(state="normal")

            SignalQualityPage(win, on_pass=on_pass).pack(fill="both", expand=True)

        styled_button(self._content, "▶ Open Signal Monitor",
                      open_sig_win, style="primary", width=26).pack(pady=14)

    def _step_calibration(self):
        tk.Label(self._content,
                 text="Please sit still and relax.\n"
                      "The system will collect a 30-second resting GSR baseline.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack(anchor="w")

        self._cal_status = tk.Label(self._content, text="Press Start when ready.",
                                     bg=BG, fg=TEXT2, font=(FONT_FAMILY, 11))
        self._cal_status.pack(pady=8)
        self._cal_bar = ttk.Progressbar(self._content, length=440,
                                          mode="determinate", maximum=30)
        self._cal_bar.pack(pady=4)
        self._next_btn.config(state="disabled")

        def start_cal():
            start_btn.config(state="disabled")
            self._cal_status.config(text="● Calibrating — please stay still.", fg=WARNING)
            self._cal_bar["value"] = 0
            self._cal_elapsed = 0
            self._run_cal()

        start_btn = styled_button(self._content, "▶ Start Calibration",
                                   start_cal, style="primary", width=24)
        start_btn.pack(pady=10)

    def _run_cal(self):
        self._cal_elapsed += 1
        self._cal_bar["value"] = self._cal_elapsed
        self._cal_status.config(text=f"● Calibrating…  {self._cal_elapsed} / 30 s")
        if self._cal_elapsed < 30:
            self._content.after(1000, self._run_cal)
        else:
            self._cal_status.config(text="✓ Calibration complete!", fg=SUCCESS)
            self._next_btn.config(state="normal")

    def _step_launch(self):
        tk.Label(self._content,
                 text="Everything is set up. Click Launch to start the WebSocket bridge\n"
                      "and begin recording. Open the Pavlovia link in a browser to start the task.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack(anchor="w")

        info = card_frame(self._content)
        info.pack(fill="x", pady=10)

        for lbl_text, val_text, val_color in [
            ("WebSocket URL:", f"ws://localhost:{WEBSOCKET_PORT}", ACCENT2),
            ("Subject ID:",    self.subject_var.get() if self.subject_var else "—", SUCCESS),
            ("Bridge script:", BRIDGE_SCRIPT, TEXT2),
        ]:
            row = tk.Frame(info, bg=BG2)
            row.pack(fill="x", padx=12, pady=5)
            tk.Label(row, text=lbl_text, bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 10, "bold"), width=16, anchor="w").pack(side="left")
            tk.Label(row, text=val_text, bg=BG2, fg=val_color,
                     font=("Courier", 10)).pack(side="left", padx=6)

    # ── Navigation ────────────────────────────────────────────────────────

    def _prev(self):
        if self._step > 0:
            self._step -= 1
            self._show_step(self._step)

    def _next(self):
        if self._step < len(self.steps) - 1:
            self._step += 1
            self._show_step(self._step)
        else:
            if self.on_complete:
                self.on_complete()


# ===========================================================================
# BRIDGE PROCESS MANAGER
# ===========================================================================

# ===========================================================================
# PAVLOVIA PANEL  — latin square URL generator
# ===========================================================================

class PavloviaPanel(tk.Frame):
    """Tab showing per-session Pavlovia URLs derived from a 4-group cyclic
    latin square, with auto group assignment and RA override."""

    def __init__(self, parent, subject_var: tk.StringVar):
        super().__init__(parent, bg=BG)
        self._subject_var = subject_var
        self._group_var   = tk.IntVar(value=1)
        self._session_var = tk.IntVar(value=1)
        self._url_labels  = {}   # task_key -> tk.Label showing URL
        self._build()
        # Auto-update whenever subject ID changes
        subject_var.trace_add("write", self._on_subject_change)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self):
        # ── Header ────────────────────────────────────────────────────────
        tk.Label(self, text="Pavlovia URL Generator", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        tk.Label(self,
                 text="4-group cyclic latin square — version rotates by one letter per group across tasks.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack(anchor="w", padx=24)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

        # ── Controls row ──────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG2, padx=14, pady=12)
        ctrl.pack(fill="x", padx=24, pady=4)

        # Subject ID display (read-only, driven by main subject_var)
        tk.Label(ctrl, text="Participant ID:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._subj_lbl = tk.Label(ctrl, text="—", bg=BG2, fg=ACCENT2,
                                    font=(FONT_FAMILY, 11, "bold"))
        self._subj_lbl.grid(row=0, column=1, sticky="w")

        # Auto-assigned group
        tk.Label(ctrl, text="Auto Group:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=2, sticky="w",
                                                       padx=(24, 8))
        self._auto_grp_lbl = tk.Label(ctrl, text="—", bg=BG2, fg=SUCCESS,
                                       font=(FONT_FAMILY, 11, "bold"))
        self._auto_grp_lbl.grid(row=0, column=3, sticky="w")

        # Override group
        tk.Label(ctrl, text="Override Group:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=4, sticky="w",
                                                       padx=(24, 8))
        grp_spin = tk.Spinbox(
            ctrl, from_=1, to=4, textvariable=self._group_var,
            width=4, font=(FONT_FAMILY, 12, "bold"),
            bg=BG3, fg=TEXT, buttonbackground=BG3,
            insertbackground=TEXT, relief="flat",
            command=self._refresh,
        )
        grp_spin.grid(row=0, column=5, sticky="w", padx=(0, 4))
        grp_spin.bind("<Return>", lambda _: self._refresh())
        grp_spin.bind("<FocusOut>", lambda _: self._refresh())

        # Session selector
        tk.Label(ctrl, text="Session:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=6, sticky="w",
                                                       padx=(24, 8))
        for i, (val, txt) in enumerate([(1, "1 — HA"), (2, "2 — EX"), (3, "3 — RR")]):
            rb = tk.Radiobutton(
                ctrl, text=txt, variable=self._session_var, value=val,
                indicatoron=False,
                bg="#d8d8d8", fg="#111111", selectcolor="#5b4fe8",
                activebackground="#c0c0c0", activeforeground="#111111",
                disabledforeground="#888888",
                font=(FONT_FAMILY, 10, "bold"),
                relief="solid", bd=1, highlightthickness=0,
                padx=10, pady=4, cursor="hand2", takefocus=0,
                command=self._refresh,
            )
            rb.grid(row=0, column=7 + i, padx=3)

        # ── Active session URL card ────────────────────────────────────────
        active_frame = tk.Frame(self, bg=BG2, padx=16, pady=14)
        active_frame.pack(fill="x", padx=24, pady=8)

        tk.Label(active_frame, text="CURRENT SESSION URL", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")

        url_row = tk.Frame(active_frame, bg=BG2)
        url_row.pack(fill="x", pady=(6, 0))

        self._active_url_var = tk.StringVar(value="—")
        url_entry = tk.Entry(
            url_row, textvariable=self._active_url_var,
            font=(FONT_FAMILY, 11), bg=BG3, fg=ACCENT2,
            insertbackground=TEXT, relief="flat", bd=4,
            state="readonly", readonlybackground=BG3,
        )
        url_entry.pack(side="left", fill="x", expand=True)

        styled_button(url_row, "Copy", self._copy_active,
                      style="primary", width=8).pack(side="left", padx=(8, 0))

        self._active_info = tk.Label(
            active_frame, text="", bg=BG2, fg=TEXT2,
            font=(FONT_FAMILY, 9))
        self._active_info.pack(anchor="w", pady=(4, 0))

        # ── All-sessions reference table ───────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=8)
        tk.Label(self, text="All Sessions — This Participant", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=24)

        tbl_outer = tk.Frame(self, bg=BG2, padx=12, pady=10)
        tbl_outer.pack(fill="x", padx=24, pady=6)

        hdrs = ["Session", "Task", "Version", "URL", ""]
        col_widths = [8, 30, 9, 60, 8]
        for col, (h, w) in enumerate(zip(hdrs, col_widths)):
            tk.Label(tbl_outer, text=h, bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 9, "bold"), width=w,
                     anchor="w").grid(row=0, column=col, padx=4, pady=2, sticky="w")
        tk.Frame(tbl_outer, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=5, sticky="ew", pady=2)

        self._tbl_rows = []
        for ti in range(3):
            row_bg = BG3 if ti % 2 == 0 else BG2
            cells = []
            for col in range(5):
                if col == 4:
                    btn = styled_button(tbl_outer, "Copy", None,
                                        style="ghost", width=6)
                    btn.grid(row=ti + 2, column=col, padx=4, pady=3)
                    cells.append(btn)
                else:
                    lbl = tk.Label(tbl_outer, text="—", bg=row_bg, fg=TEXT,
                                   font=(FONT_FAMILY, 10),
                                   anchor="w",
                                   width=col_widths[col])
                    lbl.grid(row=ti + 2, column=col, padx=4, pady=3, sticky="w")
                    cells.append(lbl)
            self._tbl_rows.append(cells)

        # ── Full latin square reference ────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=8)
        tk.Label(self, text="Latin Square Reference  (all groups)", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=24)
        self._build_ls_reference()

        # Initial refresh
        self._refresh()

    def _build_ls_reference(self):
        outer = tk.Frame(self, bg=BG2, padx=12, pady=10)
        outer.pack(fill="x", padx=24, pady=(4, 20))

        # Header
        headers = ["Group", "Participant #s",
                   "Session 1 (HA) Ver", "Session 2 (EX) Ver", "Session 3 (RR) Ver"]
        col_widths = [7, 18, 20, 20, 20]
        for col, (h, w) in enumerate(zip(headers, col_widths)):
            tk.Label(outer, text=h, bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 9, "bold"), width=w,
                     anchor="w").grid(row=0, column=col, padx=4, pady=2, sticky="w")
        tk.Frame(outer, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=5, sticky="ew", pady=2)

        for g in range(1, 5):
            row_bg = BG3 if g % 2 == 0 else BG2
            example_ids = f"...{g}, ...{g+4}, ...{g+8}"
            cells = [
                (str(g),        SUCCESS),
                (example_ids,   TEXT2),
                (f"{ls_version(g, 0)}  ({PAVLOVIA_TASKS['HA'][ls_version(g, 0)]})",  ACCENT2),
                (f"{ls_version(g, 1)}  ({PAVLOVIA_TASKS['EX'][ls_version(g, 1)]})",  ACCENT2),
                (f"{ls_version(g, 2)}  ({PAVLOVIA_TASKS['RR'][ls_version(g, 2)]})",  ACCENT2),
            ]
            for col, (text, color) in enumerate(cells):
                tk.Label(outer, text=text, bg=row_bg, fg=color,
                         font=(FONT_FAMILY, 10), anchor="w",
                         width=col_widths[col]).grid(
                    row=g + 1, column=col, padx=4, pady=4, sticky="w")

    # ── Refresh logic ─────────────────────────────────────────────────────

    def _on_subject_change(self, *_):
        subj = self._subject_var.get().strip()
        self._subj_lbl.config(text=subj or "—")
        auto_g = group_from_subject(subj) if subj else 1
        self._auto_grp_lbl.config(text=f"Group {auto_g}")
        # Only auto-update spinbox if user hasn't explicitly changed it
        # (we set it whenever subject changes so it tracks automatically)
        self._group_var.set(auto_g)
        self._refresh()

    def _refresh(self, *_):
        subj  = self._subject_var.get().strip()
        group = self._group_var.get()
        sess  = self._session_var.get()   # 1, 2, or 3
        task_idx = sess - 1

        # Clamp group
        try:
            group = max(1, min(4, int(group)))
            self._group_var.set(group)
        except (ValueError, tk.TclError):
            group = 1

        # Active URL
        ver = ls_version(group, task_idx)
        url = ls_url(group, task_idx)
        task_key = TASK_ORDER[task_idx]
        task_label = TASK_LABELS[task_key]
        exp_name   = PAVLOVIA_TASKS[task_key][ver]

        self._active_url_var.set(url)
        self._active_info.config(
            text=f"Group {group}  •  Session {sess}: {task_label}  •  Version {ver}  •  {exp_name}"
        )

        # All-sessions table
        for ti in range(3):
            v    = ls_version(group, ti)
            u    = ls_url(group, ti)
            tk_  = TASK_ORDER[ti]
            cells = self._tbl_rows[ti]
            cells[0].config(text=f"Session {ti+1}")
            cells[1].config(text=TASK_LABELS[tk_])
            cells[2].config(text=v,
                             fg=SUCCESS if ti == task_idx else ACCENT2)
            cells[3].config(text=u)
            # wire copy button
            def _copy_fn(u=u):
                self.clipboard_clear()
                self.clipboard_append(u)
            cells[4].config(command=_copy_fn)

    def _copy_active(self):
        url = self._active_url_var.get()
        if url and url != "—":
            self.clipboard_clear()
            self.clipboard_append(url)


class BridgeManager:
    def __init__(self, script_path, subject_id="subject", log_cb=None):
        self.script_path = script_path
        self.subject_id  = subject_id
        self.log_cb      = log_cb
        self._proc       = None
        self._thread     = None

    def start(self):
        if self._proc and self._proc.poll() is None:
            return False
        env = os.environ.copy()
        env["HPM_SUBJECT_ID"]    = self.subject_id
        env["PYTHONUNBUFFERED"]  = "1"   # force line-by-line stdout flush
        env["PYTHONIOENCODING"] = "utf-8"  # ensure arrow/unicode chars survive
        try:
            # Run bridge from its own directory so relative paths
            # (e.g. physiologging/) are created next to the script
            _bridge_dir = os.path.dirname(os.path.abspath(self.script_path))
            self._proc = subprocess.Popen(
                [sys.executable, "-u", self.script_path],  # -u = unbuffered
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=_bridge_dir,   # ensure physiologging/ lands next to bridge
            )
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            if self.log_cb:
                self.log_cb(f"[ERROR] Could not start bridge: {e}")
            return False

    def _read(self):
        """Read subprocess stdout line by line and forward to log callback."""
        try:
            for line in iter(self._proc.stdout.readline, ""):
                stripped = line.rstrip()
                if stripped and self.log_cb:
                    self.log_cb(stripped)
        except Exception as e:
            if self.log_cb:
                self.log_cb(f"[ERROR] Bridge reader thread: {e}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                # SIGINT triggers KeyboardInterrupt in the bridge, which
                # runs its shutdown block and calls auto_run_pipeline().
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, self._proc.pid)
                else:
                    import signal as _signal
                    os.kill(self._proc.pid, _signal.SIGINT)
            except Exception:
                self._proc.terminate()  # fallback
            try:
                self._proc.wait(timeout=15)  # pipeline may take a moment
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None


# ===========================================================================
# MAIN APP
# ===========================================================================

class HPMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HPM System")
        self.configure(bg=BG)
        self.geometry("980x700")
        self.minsize(800, 580)
        self.resizable(True, True)

        self._mode        = tk.StringVar(value="ra")
        self._subject_var = tk.StringVar(value="")
        self._bridge_mgr  = None

        self._setup_ttk_styles()
        self._build_header()
        self._build_notebook()
        self._build_status_bar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── TTK styles ────────────────────────────────────────────────────────

    def _setup_ttk_styles(self):
        s = ttk.Style(self)
        s.theme_use("default")

        # Combobox
        s.configure("TCombobox",
                    fieldbackground=BG3, background=BG3,
                    foreground=TEXT, selectbackground=ACCENT,
                    selectforeground="#ffffff", borderwidth=0,
                    arrowcolor=TEXT)

        # Progress bar
        s.configure("TProgressbar",
                    troughcolor=BG3, background=ACCENT,
                    thickness=12, borderwidth=0)

        # Notebook — high-contrast tabs
        s.configure("TNotebook",
                    background=BG, borderwidth=0,
                    tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab",
                    background=TAB_BG_INACTIVE,
                    foreground=TAB_FG_INACTIVE,
                    padding=[18, 7],
                    font=(FONT_FAMILY, 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", TAB_BG_ACTIVE),
                          ("active",   "#3a3a70")],
              foreground=[("selected", TAB_FG_ACTIVE),
                          ("active",   "#ffffff")])

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg="#0d0d20", pady=0)
        hdr.pack(fill="x")

        inner = tk.Frame(hdr, bg="#0d0d20")
        inner.pack(fill="x", padx=20, pady=10)

        # Logo
        logo_f = tk.Frame(inner, bg="#0d0d20")
        logo_f.pack(side="left")
        tk.Label(logo_f, text="HPM", bg="#0d0d20", fg=ACCENT2,
                 font=(FONT_FAMILY, 24, "bold")).pack(side="left")
        tk.Label(logo_f, text="  Psychophysiology System", bg="#0d0d20", fg=TEXT,
                 font=(FONT_FAMILY, 12)).pack(side="left", pady=4)

        # Mode toggle  — high-contrast radio buttons
        tog = tk.Frame(inner, bg="#0d0d20")
        tog.pack(side="right")
        tk.Label(tog, text="Mode:", bg="#0d0d20", fg=TEXT,
                 font=(FONT_FAMILY, 10, "bold")).pack(side="left", padx=(0, 8))

        self._mode_btns = {}
        for val, txt in [("ra", "RA Mode"), ("participant", "Participant Mode")]:
            btn = tk.Radiobutton(
                tog, text=txt, variable=self._mode, value=val,
                indicatoron=False,
                bg="#d8d8d8", fg="#111111",
                selectcolor="#5b4fe8",
                activebackground="#c0c0c0", activeforeground="#111111",
                disabledforeground="#888888",
                font=(FONT_FAMILY, 10, "bold"),
                relief="solid", bd=1, highlightthickness=0,
                padx=14, pady=6, cursor="hand2", takefocus=0,
                command=self._on_mode_change,
            )
            btn.pack(side="left", padx=3)
            self._mode_btns[val] = btn

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    # ── Notebook ──────────────────────────────────────────────────────────

    def _build_notebook(self):
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True)

        self._tab_wizard   = tk.Frame(self._nb, bg=BG)
        self._tab_monitor  = tk.Frame(self._nb, bg=BG)
        self._tab_pavlovia = tk.Frame(self._nb, bg=BG)
        self._tab_log      = tk.Frame(self._nb, bg=BG)
        self._tab_settings = tk.Frame(self._nb, bg=BG)

        self._nb.add(self._tab_wizard,   text="  Setup Wizard  ")
        self._nb.add(self._tab_monitor,  text="  Session Monitor  ")
        self._nb.add(self._tab_pavlovia, text="  Pavlovia URLs  ")
        self._nb.add(self._tab_log,      text="  Log  ")
        self._nb.add(self._tab_settings, text="  Settings  ")

        self._rebuild_wizard()
        self._build_monitor_tab()
        self._build_pavlovia_tab()
        self._build_log_tab()
        self._build_settings_tab()

    # ── Wizard tab ────────────────────────────────────────────────────────

    def _rebuild_wizard(self):
        for w in self._tab_wizard.winfo_children():
            w.destroy()
        mode  = self._mode.get()
        steps = WIZARD_STEPS_RA if mode == "ra" else WIZARD_STEPS_PARTICIPANT
        SetupWizard(
            self._tab_wizard, steps=steps, mode=mode,
            on_complete=self._on_wizard_complete,
            subject_var=self._subject_var,
        ).pack(fill="both", expand=True)

    def _on_mode_change(self):
        self._rebuild_wizard()

    def _on_wizard_complete(self):
        self._do_launch()

    # ── Pavlovia tab ──────────────────────────────────────────────────────

    def _build_pavlovia_tab(self):
        self._pavlovia_panel = PavloviaPanel(
            self._tab_pavlovia, subject_var=self._subject_var
        )
        self._pavlovia_panel.pack(fill="both", expand=True)

    # ── Monitor tab ───────────────────────────────────────────────────────

    def _build_monitor_tab(self):
        ctrl = tk.Frame(self._tab_monitor, bg=BG2, pady=8)
        ctrl.pack(fill="x", padx=14, pady=10)
        styled_button(ctrl, "▶ Launch Bridge",
                      self._do_launch, style="success", width=18).pack(side="left", padx=6)
        styled_button(ctrl, "⏹ Stop Bridge",
                      self._stop_bridge, style="danger", width=16).pack(side="left", padx=4)
        tk.Label(ctrl, text="(or complete the Setup Wizard to launch automatically)",
                 bg=BG2, fg=TEXT2, font=(FONT_FAMILY, 9)).pack(side="left", padx=10)

        self._monitor = SessionMonitor(self._tab_monitor, app_ref=self)
        self._monitor.pack(fill="both", expand=True)

    # ── Log tab ───────────────────────────────────────────────────────────

    def _build_log_tab(self):
        hdr = tk.Frame(self._tab_log, bg=BG, pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="Bridge Output Log", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 12, "bold")).pack(side="left")
        styled_button(hdr, "Clear", self._clear_log,
                      style="ghost", width=8).pack(side="right")

        self._log_text = tk.Text(
            self._tab_log, bg=BG3, fg=TEXT,
            font=("Courier", 10), relief="flat", bd=0,
            state="disabled", insertbackground=TEXT,
            selectbackground=ACCENT,
        )
        self._log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._log_text.tag_config("ok",      foreground=SUCCESS)
        self._log_text.tag_config("warn",    foreground=WARNING)
        self._log_text.tag_config("error",   foreground=DANGER)
        self._log_text.tag_config("marker",  foreground=ACCENT2)
        self._log_text.tag_config("default", foreground=TEXT)

    def _append_log(self, line):
        lo = line.lower()
        if any(w in lo for w in ("error", "fail", "✗", "traceback")):
            tag = "error"
        elif any(w in lo for w in ("warning", "⚠", "warn")):
            tag = "warn"
        elif any(w in lo for w in ("✓", "started", "saved", "complete", "connected", "▶", "⏹")):
            tag = "ok"
        elif "event:" in lo or ("frame" in lo and "→" in line) or "marker" in lo:
            tag = "marker"
        else:
            tag = "default"

        self._log_text.config(state="normal")
        self._log_text.insert("end", line + "\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

        self._route_to_monitor(line)

    # Compiled once — matches bridge's exact marker print format:
    # "[14:22:03.456] Frame 340 → EVENT: CSp.png"
    _MARKER_RE  = re.compile(r'→ EVENT:\s*(.+)', re.IGNORECASE)
    _FPS_RE     = re.compile(r'Recording FPS:\s*([\d.]+)', re.IGNORECASE)
    _FRAMES_RE  = re.compile(r'Frames captured:\s*(\d+)', re.IGNORECASE)
    _TTL_RE_MON = re.compile(r'TTL PULSE CONFIRMED', re.IGNORECASE)

    def _route_to_monitor(self, line):
        # FPS / frames
        m = self._FPS_RE.search(line)
        if m:
            self._monitor.update_stat("fps", f"{float(m.group(1)):.1f}")
        m = self._FRAMES_RE.search(line)
        if m:
            self._monitor.update_stat("frames", m.group(1))

        # Event marker  — bridge prints: "[HH:MM:SS.mmm] Frame N → EVENT: <marker>"
        m = self._MARKER_RE.search(line)
        if m:
            marker_name = m.group(1).strip()
            # Extract timestamp from beginning of line if present
            ts_m = re.match(r'\[([\d:. ]+)\]', line)
            ts = ts_m.group(1).strip() if ts_m else _now()
            self._monitor.log_marker(ts, marker_name)

        # TTL confirmation
        if self._TTL_RE_MON.search(line):
            self._monitor.increment_ttl()

        # Store the Arduino port for convenience (pre-fill port selector)
        if 'Connected to Arduino on' in line:
            import re as _re
            _pm = _re.search(r'Connected to Arduino on\s+(\S+)', line)
            if _pm:
                self._monitor._port_var.set(_pm.group(1).strip())

        # Bridge state changes
        if 'STARTING EXPERIMENT LOGGING' in line:
            self._monitor.update_stat("state", "RECORDING", SUCCESS)
        # Physio CSV path — extract from '✓ Physio log created: path'
        if 'Physio log created:' in line:
            import re as _re
            _m = _re.search(r'Physio log created:\s*(.+\.csv)', line)
            if _m:
                _csv_path = _m.group(1).strip()
                # Resolve relative path against the bridge script directory
                if not os.path.isabs(_csv_path) and self._bridge_mgr:
                    _bridge_dir = os.path.dirname(
                        os.path.abspath(self._bridge_mgr.script_path))
                    _csv_path = os.path.join(_bridge_dir, _csv_path)
                self._monitor.set_csv_path(_csv_path)
                self._append_log(f"[{_now()}] CSV path: {_csv_path}")
        elif 'EXPERIMENT END RECEIVED' in line:
            self._monitor.update_stat("state", "DONE", WARNING)
        elif 'LAUNCHING POST-ACQUISITION ANALYSIS' in line:
            self._monitor.update_stat("state", "ANALYSING", ACCENT2)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _on_bridge_log(self, line):
        # Called from subprocess reader thread — schedule onto main thread
        self.after(0, self._append_log, line)

    # ── Settings tab ──────────────────────────────────────────────────────

    def _build_settings_tab(self):
        tk.Label(self._tab_settings, text="Settings", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        tk.Label(self._tab_settings,
                 text="These values configure the bridge and pipeline scripts.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack(anchor="w", padx=24)
        tk.Frame(self._tab_settings, bg=BORDER, height=1).pack(fill="x", padx=24, pady=12)

        self._settings_vars = {}
        self._gsr_vcc_var    = tk.StringVar(value="5.0")  # 5V rail
        self._gsr_rref_var   = tk.StringVar(value="100000")  # 100kΩ for 5V CJMCU-6701
        self._gsr_adcmax_var = tk.StringVar(value="32767")  # ADS1115 16-bit default

        fields = [
            ("Participant ID",         self._subject_var,
             "Shared with the Setup Wizard."),
            ("WebSocket Port",         tk.StringVar(value=str(WEBSOCKET_PORT)),
             "Must match Pavlovia JS (default 5678)."),
            ("Target FPS",             tk.StringVar(value="60"),
             "Video recording frame rate."),
            ("Calibration Duration (s)", tk.StringVar(value="30"),
             "Resting GSR baseline length."),
            ("Log Directory",          tk.StringVar(value=LOG_DIR),
             "Folder for all session files."),
            ("GSR Vcc (V)",            self._gsr_vcc_var,
             "CJMCU-6701 supply voltage (3.3 or 5.0)."),
            ("GSR Rref (Ω)",           self._gsr_rref_var,
             "Reference resistor in ohms (default 100000)."),
            ("GSR ADC max",            self._gsr_adcmax_var,
             "ADS1115=32767 (default), 12-bit=4095, 10-bit=1023."),
        ]
        for lbl_txt, var, hint in fields:
            row = tk.Frame(self._tab_settings, bg=BG)
            row.pack(fill="x", padx=24, pady=5)
            tk.Label(row, text=lbl_txt, bg=BG, fg=TEXT,
                     font=(FONT_FAMILY, 11, "bold"), width=26, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=var, font=(FONT_FAMILY, 11),
                     bg=BG3, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=4, width=22).pack(side="left", padx=8)
            tk.Label(row, text=hint, bg=BG, fg=TEXT2,
                     font=(FONT_FAMILY, 9)).pack(side="left")
            self._settings_vars[lbl_txt] = var

        tk.Frame(self._tab_settings, bg=BORDER, height=1).pack(fill="x", padx=24, pady=14)

        # Dependency check
        tk.Label(self._tab_settings, text="Dependency Check", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=24)

        dep_frame = tk.Frame(self._tab_settings, bg=BG2, padx=12, pady=12)
        dep_frame.pack(fill="x", padx=24, pady=8)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        deps = [
            ("pyserial",      SERIAL_AVAILABLE),
            ("numpy",         NUMPY_AVAILABLE),
            ("opencv-python", CV2_AVAILABLE),
            (BRIDGE_SCRIPT,   os.path.exists(os.path.join(base_dir, BRIDGE_SCRIPT))),
            (PIPELINE_SCRIPT, os.path.exists(os.path.join(base_dir, PIPELINE_SCRIPT))),
        ]
        for col, (name, ok) in enumerate(deps):
            f = tk.Frame(dep_frame, bg=BG2, padx=6)
            f.grid(row=0, column=col, padx=8)
            tk.Label(f, text="✓" if ok else "✗",
                     fg=SUCCESS if ok else DANGER, bg=BG2,
                     font=(FONT_FAMILY, 16, "bold")).pack()
            tk.Label(f, text=name, fg=TEXT if ok else DANGER, bg=BG2,
                     font=(FONT_FAMILY, 8), wraplength=100, justify="center").pack()

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_status_bar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._sbar = StatusBar(self)
        self._sbar.pack(fill="x")
        self._sbar.set("version", "HPM  |  Bridge v5.2.2  •  Pipeline v7.17.2", TEXT2)
        self._sbar.set("bridge",  "● Bridge: IDLE", TEXT2)
        self._sbar.set("clock",   _now(), TEXT2)
        self._tick_clock()

    def _tick_clock(self):
        self._sbar.set("clock", _now())
        self.after(1000, self._tick_clock)

    # ── Bridge launch / stop ──────────────────────────────────────────────

    def _do_launch(self):
        subject = self._subject_var.get().strip()
        if not subject:
            subject = f"subj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        base_dir    = os.path.dirname(os.path.abspath(__file__))
        bridge_path = os.path.join(base_dir, BRIDGE_SCRIPT)

        if not os.path.exists(bridge_path):
            messagebox.showerror(
                "Script Not Found",
                f"Could not find:\n{bridge_path}\n\n"
                f"Place {BRIDGE_SCRIPT} in the same folder as hpm_gui.py."
            )
            return

        if self._bridge_mgr and self._bridge_mgr.is_running():
            messagebox.showinfo("Already Running",
                                "The bridge is already running.\nStop it first to restart.")
            return

        self._bridge_mgr = BridgeManager(bridge_path, subject_id=subject,
                                          log_cb=self._on_bridge_log)
        ok = self._bridge_mgr.start()
        if ok:
            # Mark bridge as active, stop ArduinoReader, hide canvases.
            self._monitor._bridge_active = True
            self._monitor._stop_monitoring()
            self._monitor.set_canvas_mode(False)
            self._sbar.set("bridge", "● Bridge: RUNNING", SUCCESS)
            self._monitor.update_stat("state", "RUNNING", SUCCESS)
            self._monitor.start_timer()
            self._nb.select(self._tab_monitor)
            self._append_log(f"[{_now()}] Bridge started  —  subject: {subject}")
        else:
            messagebox.showerror("Launch Failed",
                                  "Could not start the bridge script.\n"
                                  "Check the Log tab for details.")

    def _stop_bridge(self):
        if self._bridge_mgr:
            self._bridge_mgr.stop()
        self._monitor._bridge_active = False  # serial port released
        self._monitor.set_canvas_mode(True)   # restore canvas view
        self._monitor._mon_status.config(
            text="Bridge stopped — click Start Monitoring for live signals",
            fg=TEXT2)
        self._sbar.set("bridge", "● Bridge: STOPPED", DANGER)
        self._monitor.update_stat("state", "STOPPED", DANGER)
        self._monitor.stop_timer()
        self._append_log(f"[{_now()}] Bridge stopped.")

    # ── Close ─────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._bridge_mgr and self._bridge_mgr.is_running():
            if not messagebox.askyesno("Quit",
                                        "The bridge is still running.\nStop it and quit?"):
                return
            self._bridge_mgr.stop()
        self.after(500, self.destroy)  # brief pause for bridge shutdown


# ---------------------------------------------------------------------------
def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = HPMApp()
    app.mainloop()
