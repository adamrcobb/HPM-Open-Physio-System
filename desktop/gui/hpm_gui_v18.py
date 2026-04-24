#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPM System GUI
==============================================================================
Tkinter-based launcher and session monitor for the HPM psychophysiology system.

Two modes:
• RA Mode – full control panel for research assistants
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
# THEME — high-contrast dark palette
# ---------------------------------------------------------------------------
BG = "#1a1a2e"    # deep navy background
BG2 = "#16213e"   # card / panel background
BG3 = "#0f3460"   # input / canvas background
ACCENT = "#7c6af7"   # purple accent (buttons)
ACCENT2 = "#c4b5fd"  # light purple (values)
SUCCESS = "#4ade80"  # bright green
WARNING = "#fbbf24"  # amber
DANGER = "#f87171"   # red
TEXT = "#f0f0ff"     # near-white primary text
TEXT2 = "#c0c0d8"    # secondary text (raised for contrast)
BORDER = "#2a2a55"

# Button foregrounds — always near-black on bright BG for max contrast
BTN_LIGHT_FG = "#0a0a1a"

# Tab / menu high-contrast
TAB_BG_ACTIVE = "#7c6af7"
TAB_FG_ACTIVE = "#ffffff"
TAB_BG_INACTIVE = "#1e1e40"
TAB_FG_INACTIVE = "#c0c0d8"

FONT_FAMILY = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"

# ---------------------------------------------------------------------------
# SCRIPT NAMES (no dashes)
# ---------------------------------------------------------------------------
BRIDGE_SCRIPT = "pavlovia_arduino_bridge_v5_2_2.py"
PIPELINE_SCRIPT = "psychophysiology_pipeline_v7_17_2.py"
WEBSOCKET_PORT = 5678
ARDUINO_BAUDRATE = 115200
LOG_DIR = "physiologging"

# ---------------------------------------------------------------------------
# LONGITUDINAL ASSIGNMENTS (per-participant session history)
# ---------------------------------------------------------------------------

ASSIGNMENTS_FILE = "assignments.json"


def _assignments_path() -> str:
    """Return full path to assignments.json next to the GUI script."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, ASSIGNMENTS_FILE)


def load_assignments() -> dict:
    """Load assignments.json, returning {} if not present or invalid."""
    path = _assignments_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_assignments(data: dict) -> None:
    """Write assignments.json safely."""
    path = _assignments_path()
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        # Soft-fail; GUI should keep running even if disk write fails
        pass

# ---------------------------------------------------------------------------
# CJMCU-6701 GSR SENSOR CONVERSION
# ---------------------------------------------------------------------------
# The CJMCU-6701 outputs a raw ADC voltage; the GUI converts it to µS for display.
# Formula: Rskin (Ω) = Rref * (Vcc - Vout) / Vout
# Conductance (µS) = 1e6 / Rskin
# Tune these to match your hardware if readings look wrong.
GSR_VCC = 5.0      # Supply voltage to sensor (3.3 or 5.0 V)
GSR_RREF = 100_000 # Reference resistor in Ω — check your board (100kΩ typical for 5V systems)
GSR_ADC_MAX = 32767 # ADS1115 16-bit raw counts (change to 4095 for 12-bit, 1023 for 10-bit)


def gsr_adc_to_uS(adc: float,
                  vcc: float = GSR_VCC,
                  rref: float = GSR_RREF,
                  adc_max: float = GSR_ADC_MAX) -> float:
    """Convert CJMCU-6701 raw ADC reading to skin conductance in µS.
    Formula: Rskin = Rref * (Vcc - Vout) / Vout; conductance = 1e6 / Rskin
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
    "EX": {"A": "EX_A_final", "B": "EX_B_final",
           "C": "EX_C_final", "D": "EX_D_final"},
    "RR": {"A": "RR_A_final_v2", "B": "RR_B_final_v2",
           "C": "RR_C_final_v2", "D": "RR_D_final_v2"},
}

TASK_LABELS = {
    "HA": "Habituation / Acquisition",
    "EX": "Extinction",
    "RR": "Reinstatement / Retrieval",
}

TASK_ORDER = ["HA", "EX", "RR"]  # canonical session order
VERSIONS = ["A", "B", "C", "D"]
N_GROUPS = 4


def ls_version(group: int, task_idx: int) -> str:
    """Cyclic latin square: group 1-4, task_idx 0=HA 1=EX 2=RR."""
    return VERSIONS[(group - 1 + task_idx) % N_GROUPS]


def ls_url(group: int, task_idx: int) -> str:
    task = TASK_ORDER[task_idx]
    ver = ls_version(group, task_idx)
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
    ("Subject ID", "Enter participant ID and session notes."),
    ("Hardware Check", "Verify Arduino, webcam, and electrode connections."),
    ("Signal Quality", "Confirm clean ECG and GSR signals before starting."),
    ("Launch Experiment", "Start the bridge, then open Pavlovia in a browser."),
]

WIZARD_STEPS_PARTICIPANT = [
    ("Welcome", "Let's get you set up. This takes about 5 minutes."),
    ("Electrode Guide", "Follow the pictures to attach the sensors."),
    ("Signal Check", "We'll make sure everything is working."),
    ("Start Task", "You're ready! The experiment will begin now."),
]

# ---------------------------------------------------------------------------
# ELECTRODE GUIDE TEXT (v7 — explicit hydrogel + isotonic EDA)
# ---------------------------------------------------------------------------
ELECTRODE_TEXT = """\
ELECTRODE PLACEMENT GUIDE
──────────────────────────────────────────────────────

ECG (Heart) — 3 Conductive Adhesive Hydrogel Electrodes
┌──────────────────────────────────────────────┐
│ RED lead → Right collarbone                 │
│ YELLOW lead → Left lower ribcage           │
│ GREEN lead → Right lower abdomen           │
└──────────────────────────────────────────────┘
Tips:
• Use single-use conductive adhesive hydrogel ECG electrodes.
• Clean skin with an alcohol pad first and let it dry completely.
• Press firmly for about 10 seconds after applying each pad.
• Avoid very hairy areas — shave if needed to improve contact.
• Route leads so they do not pull on the electrodes.

EDA / GSR (Sweat) — 2 Pre-Gelled Isotonic Electrodes
┌──────────────────────────────────────────────┐
│ Place BOTH electrodes on the thenar         │
│ eminence (thumb pad) of the non-dominant    │
│ hand.                                       │
└──────────────────────────────────────────────┘
Tips:
• Use pre-gelled isotonic EDA electrodes or apply isotonic gel
  to the contact surface as instructed in the lab manual.
• Place both electrodes on the fleshy thumb-pad (thenar eminence),
  spaced slightly apart without overlapping.
• Snug but not tight — do not cause pain or restrict movement.
• Do not apply hand lotion or creams before the session.
• Allow the isotonic gel to fully wet the skin for 30–60 seconds
  before starting.

Safety:
• Remove electrodes immediately if you feel pain or burning.
• Do not place electrodes on broken or irritated skin.
• If you notice redness or itching from the adhesive, stop and
  contact the researcher.
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
        "danger": "#dc2626",
        "ghost": "#555577",
        "orange": "#ea580c",
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

def make_scrollable(parent):
    """
    Create a vertically scrollable area inside `parent`.

    Returns (outer_frame, inner_frame), where:
    - outer_frame is what you add to the Notebook tab
    - inner_frame is where you build your actual tab content
    """
    outer = tk.Frame(parent, bg=BG)

    canvas = tk.Canvas(
        outer,
        bg=BG,
        highlightthickness=0,
        borderwidth=0,
    )
    vbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    vbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas, bg=BG)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_frame_config(event):
        # Update scroll region to match content
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner.bind("<Configure>", _on_frame_config)

    def _on_canvas_config(event):
        # Keep inner frame width in sync with canvas width
        canvas.itemconfigure(window_id, width=event.width)

    canvas.bind("<Configure>", _on_canvas_config)

    # --------- mouse / trackpad scrolling bindings ---------

    root = canvas.winfo_toplevel()

    def _on_mousewheel(event):
        # Windows / macOS: event.delta is positive/negative
        delta = event.delta
        if delta == 0:
            return
        step = int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)
        canvas.yview_scroll(-step, "units")

    def _on_mousewheel_linux(event):
        # Linux: Button-4 (up) / Button-5 (down)
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    def _bind_scroll(_event):
        root.bind_all("<MouseWheel>", _on_mousewheel)
        root.bind_all("<Button-4>", _on_mousewheel_linux)
        root.bind_all("<Button-5>", _on_mousewheel_linux)

    def _unbind_scroll(_event):
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind_scroll)
    canvas.bind("<Leave>", _unbind_scroll)

    return outer, inner

# ===========================================================================
# SIGNAL CANVAS — mini oscilloscope
# ===========================================================================

class SignalCanvas(tk.Canvas):
    HISTORY = 600

    def __init__(self, parent, channel="ECG", color=SUCCESS, **kw):
        kw.setdefault("bg", BG3)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", BORDER)
        super().__init__(parent, **kw)
        self.channel = channel
        self.color = color
        self.data = []
        self._pending_redraw = False
        self._redraw_id = None  # after() handle so reset() can cancel pending redraws
        self.bind("<Configure>", self._on_resize)

    def push(self, value):
        self.data.append(value)
        if len(self.data) > self.HISTORY:
            self.data = self.data[-self.HISTORY:]
        if not self._pending_redraw:
            self._pending_redraw = True
            self._redraw_id = self.after(33, self._redraw)  # ~30 fps

    def _on_resize(self, _=None):
        self._redraw_id = self.after(50, self._redraw)

    def _redraw(self, _=None):
        self._redraw_id = None
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
            self.create_text(
                w // 2, h // 2,
                text=f"Waiting for {self.channel}…",
                fill=TEXT2, font=(FONT_FAMILY, 9)
            )
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
        self.create_text(
            8, 6, anchor="nw", text=self.channel,
            fill=TEXT2, font=(FONT_FAMILY, 9, "bold")
        )
        # current value (only shown for GSR — ECG BPM shown in stat tile instead)
        if self.channel != "ECG":
            self.create_text(
                w - 6, 6, anchor="ne",
                text=f"{pts[-1]:.1f}",
                fill=self.color, font=(FONT_FAMILY, 9, "bold")
            )

    def reset(self):
        """Clear canvas history and blank the widget. Call before each new monitoring session."""
        if self._redraw_id is not None:
            self.after_cancel(self._redraw_id)
            self._redraw_id = None
        self.data = []
        self.delete("all")
        self._pending_redraw = False
        self.after(10, self._redraw)


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
        arr = arr - arr.mean()
        # Percentile clip for BPM
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

        # Adaptive threshold
        threshold = np.percentile(np.abs(filtered), 80)
        if threshold < 1e-6:
            return None

        # Min distance 450 ms prevents double counting
        min_dist = max(1, int(fs * 0.45))
        peaks, _ = find_peaks(filtered, distance=min_dist, height=threshold)
        if len(peaks) < 3:
            return None

        rr_intervals = np.diff(peaks) / fs
        rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
        if len(rr_valid) < 2:
            return None

        bpm = 60.0 / float(np.median(rr_valid))
        if 30 <= bpm <= 200:
            return bpm
    except Exception:
        pass
    return None

# ===========================================================================
# ARDUINO READER — background thread, feeds queues
# ===========================================================================

class ArduinoReader(threading.Thread):
    """Reads serial data from Arduino and fills ECG / GSR queues."""

    _TTL_RE = re.compile(r'^T\d+$')

    def __init__(self, port, baud=115200):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.ser = None
        self.ecg_q = queue.Queue(maxsize=1000)     # filtered ECG for display
        self.ecg_raw_q = queue.Queue(maxsize=1000) # raw ECG (unused currently)
        self.gsr_q = queue.Queue(maxsize=1000)
        self.bpm_q = queue.Queue(maxsize=100)
        self.running = True
        self.connected = False
        self.error = None
        self.live_bpm = None  # most-recent BPM estimate (updated in poll)

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
                    # Arduino serial / CSV order (bridge v3.3):
                    # {ms},{rawGSR},{GSR_uS},{rawECG},{ECG_mV}[,{marker}]
                    if len(parts) >= 4:
                        try:
                            gsr_us = float(parts[2].strip())
                            ecg_raw = float(parts[3].strip())
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
# SESSION MONITOR — live dashboard
# ===========================================================================

class SessionMonitor(tk.Frame):
    def __init__(self, parent, app_ref=None):
        super().__init__(parent, bg=BG)
        self._app = app_ref
        self._elapsed = 0
        self._running = False
        self._timer_id = None
        self._ttl_count = 0
        self._gsr_ema = 0.0
        self._ecg_buf = []
        self._ecg_sos = None
        self._ecg_zi = None
        self._build()

    def _build(self):
        # Stat tiles
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=14, pady=(14, 6))

        self._stats = {}
        tiles = [
            ("state", "State", "IDLE", TEXT2),
            ("elapsed", "Elapsed", "00:00", TEXT),
            ("fps", "Video FPS", "—", TEXT),
            ("frames", "Frames", "0", TEXT),
            ("ecg_bpm", "ECG HR", "—", SUCCESS),
            ("gsr", "GSR µS", "—", ACCENT2),
            ("markers", "Markers", "0", WARNING),
            ("ttl", "TTL", "0", ACCENT2),
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
            if key == "gsr":
                self._gsr_range_lbl = tk.Label(
                    f, text="", bg=BG2, fg=TEXT2,
                    font=(FONT_FAMILY, 8, "bold")
                )
                self._gsr_range_lbl.pack()

        # Live signal canvases
        self._sig_outer = tk.Frame(self, bg=BG2, padx=8, pady=8)
        self._sig_outer.pack(fill="x", padx=14, pady=4)
        sig_outer = self._sig_outer
        tk.Label(sig_outer, text="LIVE SIGNALS", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")

        self.ecg_canvas = SignalCanvas(sig_outer, channel="ECG", color=SUCCESS, height=90)
        self.ecg_canvas.pack(fill="x", pady=(4, 2))
        self.gsr_canvas = SignalCanvas(sig_outer, channel="GSR", color=ACCENT2, height=70)
        self.gsr_canvas.pack(fill="x", pady=(2, 4))

        # Bridge-active notice
        self._bridge_notice = tk.Frame(self, bg=BG2, padx=16, pady=14)
        tk.Label(self._bridge_notice,
                 text="● Bridge Running — Signals Recording to CSV",
                 bg=BG2, fg=SUCCESS,
                 font=(FONT_FAMILY, 13, "bold")).pack()
        tk.Label(self._bridge_notice,
                 text=("Live waveforms are available after stopping the bridge.\n"
                       "ECG HR and GSR µS are updated every 2 seconds from the CSV."),
                 bg=BG2, fg=TEXT2,
                 font=(FONT_FAMILY, 10)).pack(pady=(6, 0))

        # GSR range legend
        legend = tk.Frame(self, bg=BG, padx=8)
        legend.pack(fill="x", padx=14, pady=(2, 0))
        tk.Label(legend, text="GSR range:", bg=BG, fg=TEXT2,
                 font=(FONT_FAMILY, 8)).pack(side="left", padx=(0, 6))
        for txt, col in [
            ("< 0.5 µS No signal", DANGER),
            ("0.5–3 µS Low", WARNING),
            ("3–38 µS Normal ✓", SUCCESS),
            ("38–100 µS High", WARNING),
            ("> 100 µS Artefact?", DANGER),
        ]:
            tk.Label(legend, text=txt, bg=BG, fg=col,
                     font=(FONT_FAMILY, 8, "bold")).pack(side="left", padx=8)

        # Port selector
        port_frame = tk.Frame(self, bg=BG2, padx=8, pady=6)
        port_frame.pack(fill="x", padx=14, pady=2)
        tk.Label(port_frame, text="Arduino Port for live monitoring:",
                 bg=BG2, fg=TEXT2, font=(FONT_FAMILY, 9)).pack(side="left")
        self._port_var = tk.StringVar()
        self._port_cb = ttk.Combobox(port_frame, textvariable=self._port_var,
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

        # Event marker log
        log_outer = tk.Frame(self, bg=BG2, padx=8, pady=6)
        log_outer.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        tk.Label(log_outer, text="EVENT MARKERS", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")
        self._log = tk.Text(
            log_outer, bg=BG3, fg=TEXT,
            font=("Courier", 10), relief="flat", bd=0,
            state="disabled", height=7,
            insertbackground=TEXT, selectbackground=ACCENT
        )
        self._log.pack(fill="both", expand=True, pady=(4, 0))

    # Port / monitoring controls

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
        self._ecg_sos = None
        self._ecg_zi = None
        self._ecg_buf = []
        self._gsr_ema = 0.0

    def _poll_reader(self):
        """Called every 50 ms to drain ArduinoReader queues → canvases."""
        if not self._reader or not self._reader.running:
            return

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

        # ECG filtering / BPM
        if ecg_vals and NUMPY_AVAILABLE:
            from scipy.signal import butter, sosfilt, sosfilt_zi
            if self._ecg_sos is None:
                try:
                    nyq = 125.0
                    self._ecg_sos = butter(2, [0.5/nyq, 40.0/nyq],
                                           btype='band', output='sos')
                    self._ecg_zi = sosfilt_zi(self._ecg_sos) * 0.0
                except Exception:
                    self._ecg_sos = None

            if self._ecg_sos is not None:
                chunk = np.array(ecg_vals, dtype=float)
                chunk = chunk - chunk.mean()
                if np.all(self._ecg_zi == 0) and len(chunk) > 0:
                    self._ecg_zi = sosfilt_zi(self._ecg_sos) * chunk[0]
                filtered, self._ecg_zi = sosfilt(
                    self._ecg_sos, chunk, zi=self._ecg_zi)
                for v in filtered:
                    self.ecg_canvas.push(float(v))
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
                self._ecg_buf.extend(ecg_vals)
                if len(self._ecg_buf) > 1000:
                    self._ecg_buf = self._ecg_buf[-1000:]
                buf = np.array(self._ecg_buf, dtype=float) - np.mean(self._ecg_buf)
                n_new = min(len(ecg_vals), len(buf))
                for v in buf[-n_new:]:
                    self.ecg_canvas.push(float(v))

        # GSR: ADC counts -> µS -> EMA
        for v in gsr_vals:
            try:
                _vcc = float(self._app._gsr_vcc_var.get()) if self._app else GSR_VCC
                _rref = float(self._app._gsr_rref_var.get()) if self._app else GSR_RREF
                _amax = float(self._app._gsr_adcmax_var.get()) if self._app else GSR_ADC_MAX
            except Exception:
                _vcc, _rref, _amax = GSR_VCC, GSR_RREF, GSR_ADC_MAX
            v_uS = gsr_adc_to_uS(v, vcc=_vcc, rref=_rref, adc_max=_amax)
            if v_uS > 0.0:
                if self._gsr_ema == 0.0:
                    self._gsr_ema = v_uS
                else:
                    self._gsr_ema = self._gsr_ema * 0.98 + v_uS * 0.02
                self.gsr_canvas.push(self._gsr_ema)

        if gsr_vals:
            self.update_gsr(self._gsr_ema)

        self.after(50, self._poll_reader)

    # Timer and CSV polling for bridge

    def start_timer(self):
        self._elapsed = 0
        self._running = True
        self._csv_bpm_path = None
        self._csv_file_pos = 0
        self._csv_header = None
        self._csv_ecg_buf = []
        self._csv_gsr_last = 0.0
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
        self._csv_bpm_path = path
        self._csv_file_pos = 0
        self._csv_header = None
        self._csv_ecg_buf = []
        self._ecg_buf = []
        self._ecg_sos = None
        self._ecg_zi = None
        self._gsr_ema = 0.0

    def _poll_bridge_bpm(self):
        """Every 2s: update ECG HR and GSR tile from bridge CSV."""
        if not self._running:
            return
        path = getattr(self, '_csv_bpm_path', None)
        if not path or not os.path.exists(path):
            import glob as _glob
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
                    if self._csv_header is None:
                        self._csv_header = next(_csv.reader(f), None)
                        self._csv_file_pos = f.tell()
                    else:
                        f.seek(0, 2)
                        eof = f.tell()
                        if eof <= self._csv_file_pos:
                            self.after(2000, self._poll_bridge_bpm)
                            return
                        f.seek(self._csv_file_pos)
                        for row in _csv.reader(f):
                            new_rows.append(row)
                        self._csv_file_pos = f.tell()
                header = self._csv_header
                rows = new_rows
                if not header or not rows:
                    self.after(2000, self._poll_bridge_bpm)
                    return

                # ECG
                ecg_col = next((i for i, h in enumerate(header)
                                if h in ('RawECG', 'ECG_mV', 'ECGmV')), None)
                if ecg_col is not None:
                    new_ecg = []
                    for r in rows:
                        try:
                            new_ecg.append(float(r[ecg_col]))
                        except (ValueError, IndexError):
                            pass
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

                # GSR
                gsr_col = next((i for i, h in enumerate(header)
                                if h in ('GSR_uS', 'GSRuS', 'RawGSR')), None)
                if gsr_col is not None:
                    try:
                        _vcc = float(self._app._gsr_vcc_var.get()) if self._app else GSR_VCC
                        _rref = float(self._app._gsr_rref_var.get()) if self._app else GSR_RREF
                        _amax = float(self._app._gsr_adcmax_var.get()) if self._app else GSR_ADC_MAX
                    except Exception:
                        _vcc, _rref, _amax = GSR_VCC, GSR_RREF, GSR_ADC_MAX
                    for r in rows:
                        try:
                            raw_gsr = float(r[gsr_col])
                            v_uS = gsr_adc_to_uS(raw_gsr, vcc=_vcc,
                                                 rref=_rref, adc_max=_amax)
                            if 0.05 <= v_uS <= 200.0:
                                if self._gsr_ema == 0.0:
                                    self._gsr_ema = v_uS
                                else:
                                    self._gsr_ema = self._gsr_ema * 0.98 + v_uS * 0.02
                        except (ValueError, IndexError):
                            pass
                    self.update_gsr(self._gsr_ema)
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

    # GSR physiological range
    GSR_RANGE_LOW = 0.5
    GSR_RANGE_OK = 3.0
    GSR_RANGE_HIGH = 38.0
    GSR_RANGE_MAX = 100.0

    def update_stat(self, key, text, color=None):
        if key in self._stats:
            cfg = {"text": text}
            if color:
                cfg["fg"] = color
            self._stats[key].config(**cfg)

    def set_canvas_mode(self, visible: bool):
        if visible:
            self._sig_outer.pack(fill="x", padx=14, pady=4)
            self._bridge_notice.pack_forget()
        else:
            self._sig_outer.pack_forget()
            self._bridge_notice.pack(fill="x", padx=14, pady=8)

    def update_gsr(self, us_value: float):
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
        self._log.insert("end", f"[{ts}] {marker}\n")
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
# SIGNAL QUALITY PAGE (popup used in wizard)
# ===========================================================================

class SignalQualityPage(tk.Frame):
    def __init__(self, parent, on_pass=None):
        super().__init__(parent, bg=BG)
        self.on_pass = on_pass
        self.reader = None
        self._ecg_ok = False
        self._gsr_ok = False
        self._ecg_buf = []
        self._gsr_ema = 0.0
        self._ecg_sos = None
        self._ecg_zi = None
        self._poll_id = None
        self._gen = 0
        self._build()

    def _build(self):
        tk.Label(self, text="Signal Quality Check", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 16, "bold")).pack(pady=(18, 2))
        tk.Label(self,
                 text="Confirm clean ECG and GSR signals before proceeding.",
                 bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)).pack()

        # Port row
        sel = card_frame(self)
        sel.pack(fill="x", padx=20, pady=10)
        port_row = tk.Frame(sel, bg=BG2)
        port_row.pack(fill="x", padx=10, pady=8)
        tk.Label(port_row, text="Port:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self._port_var = tk.StringVar()
        self._port_cb = ttk.Combobox(port_row, textvariable=self._port_var,
                                     state="readonly", width=24,
                                     font=(FONT_FAMILY, 10))
        self._port_cb.pack(side="left", padx=6)
        styled_button(port_row, "↻ Refresh", self._refresh,
                      style="ghost", width=10).pack(side="left", padx=4)
        styled_button(port_row, "▶ Connect", self._connect,
                      style="primary", width=10).pack(side="left")
        self._conn_lbl = tk.Label(port_row, text="", bg=BG2, fg=TEXT2,
                                  font=(FONT_FAMILY, 9))
        self._conn_lbl.pack(side="left", padx=8)
        self.after(150, self._refresh)

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

        # Continue button
        self._cont_btn = styled_button(self, "Continue →", self._continue,
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
        f._dot = dot
        f._ok = False
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
            self.reader = None

        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None

        self._ecg_buf = []
        self._gsr_ema = 0.0
        self._ecg_sos = None
        self._ecg_zi = None
        self.ecg_cv.reset()
        self.gsr_cv.reset()

        self._ecg_ind._dot.config(fg=TEXT2)
        self._ecg_ind._ok = False
        self._ecg_ind.winfo_children()[2].config(text="Waiting…", fg=TEXT2)
        self._gsr_ind._dot.config(fg=TEXT2)
        self._gsr_ind._ok = False
        self._gsr_ind.winfo_children()[2].config(text="Waiting…", fg=TEXT2)
        self._cont_btn.config(state="disabled")

        self._gen += 1
        my_gen = self._gen

        self.reader = ArduinoReader(port, ARDUINO_BAUDRATE)
        self.reader.connected = False  # NEW line
        self.reader.start()
        self._conn_lbl.config(text=f"Connecting to {port}…", fg=WARNING)
        self._poll_id = self.after(80, lambda: self._poll(my_gen))

    def _poll(self, gen=None):
        self._poll_id = None

        # If a newer connect has happened, discard this poll
        if gen is not None and gen != self._gen:
            return

        if not self.reader or not self.reader.running:
            return

        if self.reader.error:
            self._conn_lbl.config(text=f"Error: {self.reader.error}", fg=DANGER)
            return

        if self.reader.connected and "Connecting" in self._conn_lbl.cget("text"):
            self._conn_lbl.config(text=f"● {self.reader.port}", fg=SUCCESS)

        # Drain new samples from reader
        ecg = self.reader.drain_ecg(60)
        gsr = self.reader.drain_gsr(60)

        # ECG buffer
        self._ecg_buf.extend(ecg)
        if len(self._ecg_buf) > 1000:
            self._ecg_buf = self._ecg_buf[-1000:]

        # ECG filtering and drawing
        if ecg and NUMPY_AVAILABLE:
            from scipy.signal import butter, sosfilt, sosfilt_zi

            # Initialize filter on first use after reset
            if self._ecg_sos is None:
                try:
                    nyq = 125.0
                    self._ecg_sos = butter(
                        2, [0.5 / nyq, 40.0 / nyq],
                        btype="band", output="sos"
                    )
                    self._ecg_zi = None
                except Exception:
                    self._ecg_sos = None

            if self._ecg_sos is not None:
                chunk = np.array(ecg, dtype=float)
                chunk = chunk - chunk.mean()

                if self._ecg_zi is None:
                    if len(chunk) > 0:
                        self._ecg_zi = sosfilt_zi(self._ecg_sos) * chunk[0]
                    else:
                        self._ecg_zi = sosfilt_zi(self._ecg_sos) * 0.0

                filtered, self._ecg_zi = sosfilt(
                    self._ecg_sos, chunk, zi=self._ecg_zi
                )
                for v in filtered:
                    self.ecg_cv.push(float(v))
        else:
            # Fallback: just plot DC‑removed buffer
            if self._ecg_buf:
                buf = np.array(self._ecg_buf, dtype=float)
                buf -= buf.mean()
                n_new = min(len(ecg), len(buf))
                for v in buf[-n_new:]:
                    self.ecg_cv.push(float(v))

        # GSR smoothing and drawing
        for v in gsr:
            if 0.05 <= v <= 200.0:
                if self._gsr_ema == 0.0:
                    self._gsr_ema = v
                else:
                    self._gsr_ema = self._gsr_ema * 0.98 + v * 0.02
                self.gsr_cv.push(self._gsr_ema)

        # Quality heuristics
        if NUMPY_AVAILABLE and len(self._ecg_buf) >= 50:
            buf = np.array(self._ecg_buf, dtype=float)
            buf = buf - buf.mean()
            ecg_std = float(np.std(buf))
            ecg_range = float(buf.max() - buf.min())
            signal_ok = (
                ecg_range > 0
                and ecg_std > 0.01 * ecg_range
                and ecg_std > 0.05
            )
            bpm = _estimate_bpm(buf, fs=250.0)
            if signal_ok and bpm is not None:
                self._ecg_ind._dot.config(fg=SUCCESS)
                self._ecg_ind._ok = True
                self._ecg_ind.winfo_children()[2].config(
                    text=f"{bpm:.0f} BPM", fg=SUCCESS
                )
            elif signal_ok:
                self._ecg_ind._dot.config(fg=WARNING)
                self._ecg_ind.winfo_children()[2].config(
                    text="Signal OK — detecting…", fg=WARNING
                )
                self._ecg_ind._ok = False
            else:
                self._ecg_ind._dot.config(fg=DANGER)
                self._ecg_ind.winfo_children()[2].config(
                    text="Weak signal", fg=DANGER
                )
                self._ecg_ind._ok = False

        if len(gsr) >= 5:
            gsr_mean = float(np.mean(gsr)) if NUMPY_AVAILABLE else sum(gsr) / len(gsr)
            if gsr_mean > 0.1:
                self._gsr_ind._dot.config(fg=SUCCESS)
                self._gsr_ind._ok = True
                self._gsr_ind.winfo_children()[2].config(
                    text=f"OK ({gsr_mean:.2f} µS)", fg=SUCCESS
                )
            else:
                self._gsr_ind._dot.config(fg=WARNING)
                self._gsr_ind.winfo_children()[2].config(
                    text="Check contacts", fg=WARNING
                )

        both = self._ecg_ind._ok and self._gsr_ind._ok
        self._cont_btn.config(state="normal" if both else "disabled")

        # Schedule next poll for this generation
        self._poll_id = self.after(80, lambda: self._poll(gen))
        
    def _continue(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        if self.on_pass:
            self.on_pass()
            
# ===========================================================================
# SETUP WIZARD
# ===========================================================================

class SetupWizard(tk.Frame):
    def __init__(
        self,
        parent,
        steps,
        mode="ra",
        on_complete=None,
        subject_var=None,
        session_var=None,
        version_var=None,
        app_ref=None,
    ):
        super().__init__(parent, bg=BG)
        self.steps = steps
        self.mode = mode
        self.on_complete = on_complete
        self.subject_var = subject_var
        self.session_var = session_var      # NEW
        self.version_var = version_var      # NEW
        self._app_ref = app_ref             # NEW (HPMApp reference)

        self._step = 0
        self._cal_elapsed = 0
        self._build()
        self._show_step(0)
        
    def _build(self):
        # Progress dots
        self._strip = tk.Frame(self, bg=BG, pady=10)
        self._strip.pack(fill="x", padx=20)
        self._dots = []
        for i, (title, _) in enumerate(self.steps):
            col = tk.Frame(self._strip, bg=BG)
            col.pack(side="left", expand=True)
            dot = tk.Label(col, text="●", fg=BORDER, bg=BG,
                           font=(FONT_FAMILY, 16))
            dot.pack()
            tk.Label(col, text=title, fg=TEXT2, bg=BG,
                     font=(FONT_FAMILY, 8), wraplength=80,
                     justify="center").pack()
            self._dots.append(dot)

        separator(self).pack(fill="x", padx=20)

        # Content
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True, padx=20, pady=10)

        # Nav
        nav = tk.Frame(self, bg=BG, pady=10)
        nav.pack(fill="x", padx=20)
        self._back_btn = styled_button(nav, "← Back", self._prev,
                                       style="ghost", width=12)
        self._back_btn.pack(side="left")
        self._next_btn = styled_button(nav, "Next →", self._next,
                                       style="primary", width=14)
        self._next_btn.pack(side="right")

    def _clear_content(self):
        for w in self._content.winfo_children():
            w.destroy()

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

    # Step builders

    def _step_subject(self):
        if self.mode == "participant":
            tk.Label(
                self._content,
                text=("Welcome to the HPM Study!\n\n"
                      "This session will record your heart rate and skin\n"
                      "conductance while you complete a short computer task.\n\n"
                      "Enter your Participant ID below (given by your researcher)."),
                bg=BG, fg=TEXT, font=(FONT_FAMILY, 11),
                justify="left"
            ).pack(anchor="w")
        else:
            tk.Label(self._content, text="Session Setup", bg=BG, fg=TEXT,
                     font=(FONT_FAMILY, 13, "bold")).pack(anchor="w")

        frm = card_frame(self._content)
        frm.pack(fill="x", pady=10)

        row = tk.Frame(frm, bg=BG2)
        row.pack(fill="x", padx=12, pady=10)
        tk.Label(row, text="Participant ID:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold"), width=16,
                 anchor="w").pack(side="left")
        ent = tk.Entry(row, textvariable=self.subject_var,
                       font=(FONT_FAMILY, 12), bg=BG3, fg=TEXT,
                       insertbackground=TEXT, relief="flat", bd=4, width=20)
        ent.pack(side="left", padx=8)
        ent.focus_set()

        if self.mode == "ra":
            row2 = tk.Frame(frm, bg=BG2)
            row2.pack(fill="x", padx=12, pady=(0, 4))
            tk.Label(row2, text="Session Notes:", bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 11, "bold"), width=16,
                     anchor="w").pack(side="left", anchor="n")
            self._notes = tk.Text(
                frm, height=3, bg=BG3, fg=TEXT, font=(FONT_FAMILY, 11),
                relief="flat", bd=4, insertbackground=TEXT
            )
            self._notes.pack(fill="x", padx=12, pady=(0, 10))

    def _step_hardware(self):
        items = [
            ("Arduino", "USB cable plugged in — green power LED on."),
            ("Webcam", "USB webcam connected and positioned at face level."),
            (
                "Electrodes",
                "ECG leads with conductive adhesive hydrogel electrodes and "
                "EDA leads with pre-gelled isotonic electrodes on the thenar "
                "eminence of the non-dominant hand."
            ),
            ("Faraday Cage", "Lid secured on enclosure (copper-lined box)."),
        ]
        # rest of method unchanged
        
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
                text=f" {key}", anchor="w",
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
        tk.Label(
            self._content,
            text=("Click the button below to open the live signal monitor.\n"
                  "Confirm both ECG and GSR show clean traces, then close it to continue."),
            bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)
        ).pack(anchor="w")
        self._next_btn.config(state="disabled")

        def open_sig_win():
            win = tk.Toplevel(self._content)
            win.title("Signal Quality Check")
            win.configure(bg=BG)
            win.geometry("660x540")

            def on_pass():
                win.destroy()
                self._next_btn.config(state="normal")

            page = SignalQualityPage(win, on_pass=on_pass)
            page.pack(fill="both", expand=True)

            def on_close():
                if page.reader:
                    page.reader.stop()
                    page.reader = None
                if page._poll_id is not None:
                    page.after_cancel(page._poll_id)
                    page._poll_id = None
                win.destroy()

            win.protocol("WM_DELETE_WINDOW", on_close)

        styled_button(self._content, "▶ Open Signal Monitor",
                      open_sig_win, style="primary", width=26).pack(pady=14)
        
    def _step_calibration(self):
        tk.Label(
            self._content,
            text=("Please sit still and relax.\n"
                  "The system will collect a 30-second resting GSR baseline."),
            bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)
        ).pack(anchor="w")

        self._cal_status = tk.Label(
            self._content, text="Press Start when ready.",
            bg=BG, fg=TEXT2, font=(FONT_FAMILY, 11)
        )
        self._cal_status.pack(pady=8)
        self._cal_bar = ttk.Progressbar(
            self._content, length=440,
            mode="determinate", maximum=30
        )
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
        self._cal_status.config(text=f"● Calibrating… {self._cal_elapsed} / 30 s")
        if self._cal_elapsed < 30:
            self._content.after(1000, self._run_cal)
        else:
            self._cal_status.config(text="✓ Calibration complete!", fg=SUCCESS)
            self._next_btn.config(state="normal")

    def _step_launch(self):
        """Guided experiment-launch checklist with inline bridge start and Pavlovia URL."""
        import webbrowser as _wb

        self._next_btn.config(state="disabled")

        subj = self.subject_var.get().strip() if self.subject_var else "—"
        grp = group_from_subject(subj) if subj and subj != "—" else 1

        # Load or initialize assignment record for this subject
        assignments = self._app_ref._assignments if self._app_ref else {}
        rec = assignments.get(subj, {"group": grp, "sessions_completed": {}})
        sessions_completed = rec.get("sessions_completed", {})

        # Helper: next recommended session/version based on what is completed
        def _recommended_session_and_version():
            for idx, task in enumerate(TASK_ORDER):
                if task not in sessions_completed:
                    ver = ls_version(grp, idx)
                    return task, ver
            last_task = TASK_ORDER[-1]
            last_idx = len(TASK_ORDER) - 1
            return last_task, ls_version(grp, last_idx)

        # Subject / session info card
        info = card_frame(self._content)
        info.pack(fill="x", pady=(0, 8))
        for lbl_text, val_text, val_color in [
            ("Subject ID:", subj or "—", SUCCESS),
            ("Group:", str(grp), ACCENT2),
            ("Bridge script:", BRIDGE_SCRIPT, TEXT2),
            ("WebSocket:", f"ws://localhost:{WEBSOCKET_PORT}", ACCENT2),
        ]:
            row = tk.Frame(info, bg=BG2)
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=lbl_text, bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 10, "bold"),
                     width=16, anchor="w").pack(side="left")
            tk.Label(row, text=val_text, bg=BG2, fg=val_color,
                     font=("Courier", 10)).pack(side="left", padx=4)

        # ------------------------------------------------------------------
        # Session + Version selection with longitudinal recommendations
        # ------------------------------------------------------------------
        select_card = card_frame(self._content)
        select_card.pack(fill="x", pady=(4, 8))
        tk.Label(
            select_card,
            text="Session and Version",
            bg=BG2, fg=TEXT,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 2))

        row = tk.Frame(select_card, bg=BG2)
        row.pack(fill="x", padx=12, pady=4)

        # Ensure we have backing vars
        if self.session_var is None:
            self.session_var = tk.StringVar(value="HA")
        if self.version_var is None:
            self.version_var = tk.StringVar(value="A")

        # Initialize from longitudinal recommendation if possible
        rec_sess, rec_ver = _recommended_session_and_version()
        if not self.session_var.get() or self.session_var.get() not in TASK_ORDER:
            self.session_var.set(rec_sess)
        if not self.version_var.get() or self.version_var.get() not in VERSIONS:
            self.version_var.set(rec_ver)

        # Session combobox
        tk.Label(row, text="Session:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 10, "bold"),
                 width=12, anchor="w").pack(side="left")
        session_cb = ttk.Combobox(
            row,
            textvariable=self.session_var,
            values=[f"{code} — {TASK_LABELS[code]}" for code in TASK_ORDER],
            state="readonly",
            width=28,
        )
        session_cb.pack(side="left", padx=6)

        # Version combobox
        row2 = tk.Frame(select_card, bg=BG2)
        row2.pack(fill="x", padx=12, pady=(2, 4))

        tk.Label(row2, text="Version:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 10, "bold"),
                 width=12, anchor="w").pack(side="left")
        ver_cb = ttk.Combobox(
            row2,
            textvariable=self.version_var,
            values=VERSIONS,
            state="readonly",
            width=6,
        )
        ver_cb.pack(side="left", padx=6)

        # Completed + next recommended label
        rec_row = tk.Frame(select_card, bg=BG2)
        rec_row.pack(fill="x", padx=12, pady=(0, 8))
        self._rec_label = tk.Label(
            rec_row,
            text="",
            bg=BG2,
            fg=TEXT2,
            font=(FONT_FAMILY, 9),
            anchor="w",
        )
        self._rec_label.pack(side="left")

        def _update_recommendation():
            # Completed summary
            if sessions_completed:
                completed_str = ", ".join(
                    f"{TASK_LABELS[s]} (v{v})"
                    for s, v in sessions_completed.items()
                    if s in TASK_LABELS
                )
                completed_part = f"Completed: {completed_str}. "
            else:
                completed_part = "Completed: none yet. "

            # Next recommended based on *uncompleted* sessions
            next_sess, next_ver = _recommended_session_and_version()
            next_part = (
                f"Latin-square recommended next: "
                f"{TASK_LABELS[next_sess]} (v{next_ver})."
            )

            self._rec_label.config(text=completed_part + next_part)
            
        def _normalize_session(*_):
            val = self.session_var.get()
            for code in TASK_ORDER:
                if code in val:
                    self.session_var.set(code)
                    break
            _update_recommendation()

        session_cb.bind("<<ComboboxSelected>>", _normalize_session)

        # Initialize label
        _update_recommendation()

        # NEW: note about longitudinal tracking
        note_row = tk.Frame(select_card, bg=BG2)
        note_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            note_row,
            text="Note: You must click 'Open Pavlovia' on this page to log task "
                 "completion for longitudinal tracking.",
            bg=BG2,
            fg=TEXT2,
            font=(FONT_FAMILY, 8),
            wraplength=520,
            justify="left",
        ).pack(anchor="w")
        
        # ------------------------------------------------------------------
        # Launch checklist (updated electrodes line)
        # ------------------------------------------------------------------
        steps_outer = card_frame(self._content)
        steps_outer.pack(fill="x", pady=4)
        tk.Label(steps_outer, text="LAUNCH CHECKLIST", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(8, 2))

        self._launch_vars = {}
        self._launch_btns = {}

        step_defs = [
            ("consent", "Informed consent confirmed",
             "Verbal or written consent obtained before proceeding."),
            ("room", "Room & participant ready",
             "Quiet, distraction-free room. Participant seated comfortably."),
            ("recorder", "Video recorder connected",
             "Connect Elgato Facecam into USB hub connected to laptop."),
            ("datacable", "Data cable connected",
             "Connect data cable with USB power blocker to USB hub."),
            (
                "electrodes", "Electrode leads connected",
                "Connect ECG leads with conductive adhesive hydrogel electrodes "
                "and EDA leads with pre-gelled isotonic electrodes on the thenar "
                "eminence of the non-dominant hand."
            ),
            ("bridge", "Bridge started",
             "WebSocket bridge must be running before Pavlovia connects.",
             "▶ Start Bridge"),
            ("pavlovia", "Pavlovia experiment opened in browser",
             "Open the selected session/version URL for this participant. "
             "Clicking this button also records the session as completed for longitudinal tracking.",
             "🌐 Open Pavlovia"),
            ("running", "Experiment confirmed running on screen",
             "Verify the task has loaded and the first screen is visible."),
        ]

        def _get_app():
            w = self._content
            while w:
                if hasattr(w, "_do_launch") and hasattr(w, "_bridge_mgr"):
                    return w
                try:
                    w = w.master
                except Exception:
                    break
            return None

        def _on_check(*_):
            ok = all(v.get() for v in self._launch_vars.values())
            self._next_btn.config(state="normal" if ok else "disabled")

        def _make_action(key, btn_label):
            def _cmd():
                if key == "bridge":
                    app = self._app_ref
                    if app:
                        app._do_launch()

                        def _check_bridge(n=0):
                            if app._bridge_mgr and app._bridge_mgr.is_running():
                                self._launch_vars["bridge"].set(True)
                                self._launch_btns["bridge"].config(
                                    text="✓ Bridge Running", state="disabled",
                                    bg="#d8d8d8", fg="#228822",
                                )
                            elif n < 20:
                                self._content.after(500, lambda: _check_bridge(n + 1))
                            else:
                                self._launch_vars["bridge"].set(True)
                                messagebox.showinfo(
                                    "Bridge",
                                    "Launch the bridge from the RA Control tab, then return here."
                                )

                        self._content.after(500, _check_bridge)

                elif key == "pavlovia":
                    # Use session/version from wizard
                    sess = self.session_var.get()
                    if sess not in TASK_ORDER:
                        sess = TASK_ORDER[0]
                    ver = self.version_var.get()
                    if ver not in VERSIONS:
                        ver = ls_version(grp, TASK_ORDER.index(sess))

                    exp_name = PAVLOVIA_TASKS[sess][ver]
                    url = PAVLOVIA_BASE + exp_name.lower()

                    # Update longitudinal assignments
                    if self._app_ref:
                        subj_cur = self.subject_var.get().strip() if self.subject_var else ""
                        if subj_cur:
                            assignments = self._app_ref._assignments
                            rec = assignments.get(
                                subj_cur,
                                {"group": grp, "sessions_completed": {}}
                            )
                            rec["group"] = grp
                            sc = rec.get("sessions_completed", {})
                            sc[sess] = ver
                            rec["sessions_completed"] = sc
                            assignments[subj_cur] = rec
                            self._app_ref._assignments = assignments
                            save_assignments(assignments)
                            _update_recommendation()

                    # Open selected URL
                    try:
                        import webbrowser as _wb_inner
                        _wb_inner.open(url)
                        self._launch_vars["pavlovia"].set(True)
                        self._launch_btns["pavlovia"].config(
                            text="✓ Opened", state="disabled",
                            bg="#d8d8d8", fg="#228822"
                        )
                    except Exception as e:
                        messagebox.showerror("Pavlovia", f"Could not open browser:\n{e}")

            return _cmd

        # Small helper window with details (your existing code can stay)

        for item in step_defs:
            key = item[0]
            label_txt = item[1]
            detail = item[2]
            has_btn = len(item) > 3

            row = tk.Frame(steps_outer, bg=BG2)
            row.pack(fill="x", padx=12, pady=5)

            var = tk.BooleanVar()
            self._launch_vars[key] = var
            var.trace_add("write", _on_check)

            cb = tk.Checkbutton(
                row, variable=var, bg=BG2,
                activebackground=BG2, selectcolor="#3a3a80",
                fg=TEXT, font=(FONT_FAMILY, 11, "bold"),
                text=f" {label_txt}", anchor="w",
                highlightthickness=0, takefocus=0,
            )
            cb.pack(side="left")

            if has_btn:
                btn_label = item[3]
                btn = styled_button(
                    row, btn_label,
                    _make_action(key, btn_label),
                    style="primary", width=20,
                )
                btn.pack(side="right", padx=6)
                self._launch_btns[key] = btn

            tk.Label(row, text=f"— {detail}", bg=BG2, fg=TEXT2,
                     font=(FONT_FAMILY, 9)).pack(side="left", padx=6)

        # Latin-square reference box can remain, or you can delete it if redundant
        url_ref = card_frame(self._content)
        url_ref.pack(fill="x", pady=(6, 2))
        tk.Label(
            url_ref,
            text="PAVLOVIA URLS (Latin-square reference)",
            bg=BG2, fg=TEXT,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 2))

        import webbrowser as _wb2
        for i, task in enumerate(TASK_ORDER):
            url = ls_url(grp, i)
            task_label = TASK_LABELS[task]
            row = tk.Frame(url_ref, bg=BG2)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(
                row, text=f"{task_label}:", bg=BG2, fg=TEXT2,
                font=(FONT_FAMILY, 9, "bold"),
                width=30, anchor="w",
            ).pack(side="left")
            lnk = tk.Label(
                row, text=url, bg=BG2, fg=ACCENT2,
                font=("Courier", 9), cursor="hand2",
            )
            lnk.pack(side="left", padx=4)
            lnk.bind("<Button-1>", lambda e, u=url: _wb2.open(u))
        tk.Label(
            url_ref,
            text="(click any URL to open in browser)",
            bg=BG2, fg=TEXT2,
            font=(FONT_FAMILY, 8),
        ).pack(anchor="w", padx=12, pady=(0, 8))
        
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
# PAVLOVIA PANEL — latin square URL generator
# (unchanged from v6 except it reads group_from_subject / ls_url)
# ===========================================================================

class PavloviaPanel(tk.Frame):
    """Tab showing per-session Pavlovia URLs derived from a 4-group cyclic
    latin square, with auto group assignment and RA override.
    """
    def __init__(self, parent, subject_var: tk.StringVar):
        super().__init__(parent, bg=BG)
        self._subject_var = subject_var
        self._group_var = tk.IntVar(value=1)
        self._session_var = tk.IntVar(value=1)
        self._url_labels = {}
        self._build()
        subject_var.trace_add("write", self._on_subject_change)

    def _build(self):
        tk.Label(self, text="Pavlovia URL Generator", bg=BG, fg=TEXT,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        tk.Label(
            self,
            text="4-group cyclic latin square — version rotates by one letter per group across tasks.",
            bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10)
        ).pack(anchor="w", padx=24)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

        # Controls
        ctrl = tk.Frame(self, bg=BG2, padx=14, pady=12)
        ctrl.pack(fill="x", padx=24, pady=4)

        tk.Label(ctrl, text="Participant ID:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=0,
                                                      sticky="w", padx=(0, 8))
        self._subj_lbl = tk.Label(ctrl, text="—", bg=BG2, fg=ACCENT2,
                                  font=(FONT_FAMILY, 11, "bold"))
        self._subj_lbl.grid(row=0, column=1, sticky="w")

        tk.Label(ctrl, text="Auto Group:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=2,
                                                      sticky="w", padx=(24, 8))
        self._auto_grp_lbl = tk.Label(ctrl, text="—", bg=BG2, fg=SUCCESS,
                                      font=(FONT_FAMILY, 11, "bold"))
        self._auto_grp_lbl.grid(row=0, column=3, sticky="w")

        tk.Label(ctrl, text="Override Group:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=4,
                                                      sticky="w", padx=(24, 8))
        grp_spin = tk.Spinbox(
            ctrl, from_=1, to=4, textvariable=self._group_var,
            width=4, font=(FONT_FAMILY, 12, "bold"),
            bg=BG3, fg=TEXT, buttonbackground=BG3,
            insertbackground=TEXT, relief="flat",
            command=self._refresh,
        )
        grp_spin.grid(row=0, column=5, sticky="w", padx=(0, 4))
        grp_spin.bind("<FocusOut>", lambda _: self._refresh())
        grp_spin.bind("<Return>", lambda _: self._refresh())

        tk.Label(ctrl, text="Session:", bg=BG2, fg=TEXT,
                 font=(FONT_FAMILY, 11, "bold")).grid(row=0, column=6,
                                                      sticky="w", padx=(24, 8))
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

        # Active URL card
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
            font=(FONT_FAMILY, 9)
        )
        self._active_info.pack(anchor="w", pady=(4, 0))

        # All sessions reference
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
                     anchor="w").grid(row=0, column=col,
                                      padx=4, pady=2, sticky="w")
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
                    lbl.grid(row=ti + 2, column=col, padx=4, pady=3,
                             sticky="w")
                    cells.append(lbl)
            self._tbl_rows.append(cells)

        # Latin square reference (omitted here for brevity – same as v6)

        self._build_ls_reference()
        self._refresh()

    def _build_ls_reference(self):
        outer = tk.Frame(self, bg=BG2, padx=12, pady=10)
        outer.pack(fill="x", padx=24, pady=(4, 20))

        headers = ["Group", "Participant #s",
                   "Session 1 (HA) Ver", "Session 2 (EX) Ver", "Session 3 (RR) Ver"]
        col_widths = [7, 18, 20, 20, 20]
        for col, (h, w) in enumerate(zip(headers, col_widths)):
            tk.Label(outer, text=h, bg=BG2, fg=TEXT,
                     font=(FONT_FAMILY, 9, "bold"), width=w,
                     anchor="w").grid(row=0, column=col,
                                      padx=4, pady=2, sticky="w")
        tk.Frame(outer, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=5, sticky="ew", pady=2)

        for g in range(1, 5):
            row_bg = BG3 if g % 2 == 0 else BG2
            example_ids = f"...{g}, ...{g+4}, ...{g+8}"
            cells = [
                (str(g), SUCCESS),
                (example_ids, TEXT2),
                (f"{ls_version(g, 0)} ({PAVLOVIA_TASKS['HA'][ls_version(g, 0)]})", ACCENT2),
                (f"{ls_version(g, 1)} ({PAVLOVIA_TASKS['EX'][ls_version(g, 1)]})", ACCENT2),
                (f"{ls_version(g, 2)} ({PAVLOVIA_TASKS['RR'][ls_version(g, 2)]})", ACCENT2),
            ]
            for col, (text, color) in enumerate(cells):
                tk.Label(outer, text=text, bg=row_bg, fg=color,
                         font=(FONT_FAMILY, 10), anchor="w",
                         width=col_widths[col]).grid(
                    row=g + 1, column=col, padx=4, pady=4, sticky="w"
                )

    def _on_subject_change(self, *_):
        subj = self._subject_var.get().strip()
        self._subj_lbl.config(text=subj or "—")
        auto_g = group_from_subject(subj) if subj else 1
        self._auto_grp_lbl.config(text=f"Group {auto_g}")
        self._group_var.set(auto_g)
        self._refresh()

    def _refresh(self, *_):
        subj = self._subject_var.get().strip()
        group = self._group_var.get()
        sess = self._session_var.get()
        task_idx = sess - 1

        try:
            group = max(1, min(4, int(group)))
            self._group_var.set(group)
        except (ValueError, tk.TclError):
            group = 1

        ver = ls_version(group, task_idx)
        url = ls_url(group, task_idx)
        task_key = TASK_ORDER[task_idx]
        task_label = TASK_LABELS[task_key]
        exp_name = PAVLOVIA_TASKS[task_key][ver]

        self._active_url_var.set(url)
        self._active_info.config(
            text=f"Group {group} • Session {sess}: {task_label} • Version {ver} • {exp_name}"
        )

        for ti in range(3):
            v = ls_version(group, ti)
            u = ls_url(group, ti)
            tk_ = TASK_ORDER[ti]
            cells = self._tbl_rows[ti]
            cells[0].config(text=f"Session {ti+1}")
            cells[1].config(text=TASK_LABELS[tk_])
            cells[2].config(text=v, fg=SUCCESS if ti == task_idx else ACCENT2)
            cells[3].config(text=u)

            def _copy_fn(u=u):
                self.clipboard_clear()
                self.clipboard_append(u)
            cells[4].config(command=_copy_fn)

    def _copy_active(self):
        url = self._active_url_var.get()
        if url and url != "—":
            self.clipboard_clear()
            self.clipboard_append(url)

# ===========================================================================
# BRIDGE PROCESS MANAGER
# ===========================================================================

class BridgeManager:
    def __init__(self, script_path, subject_id="subject", log_cb=None):
        self.script_path = script_path
        self.subject_id = subject_id
        self.log_cb = log_cb
        self._proc = None
        self._thread = None

    def start(self):
        if self._proc and self._proc.poll() is None:
            return False
        env = os.environ.copy()
        env["HPM_SUBJECT_ID"] = self.subject_id
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            _bridge_dir = os.path.dirname(os.path.abspath(self.script_path))
            self._proc = subprocess.Popen(
                [sys.executable, "-u", self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=_bridge_dir,
            )
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            if self.log_cb:
                self.log_cb(f"[ERROR] Could not start bridge: {e}")
            return False

    def _read(self):
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
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, self._proc.pid)
                else:
                    import signal as _signal
                    os.kill(self._proc.pid, _signal.SIGINT)
            except Exception:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
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

        self._mode = tk.StringVar(value="ra")
        self._subject_var = tk.StringVar(value="")

        # Session / version state and longitudinal assignments
        self._session_var = tk.StringVar(value="HA")   # HA / EX / RR
        self._version_var = tk.StringVar(value="A")    # A / B / C / D
        self._assignments = load_assignments()

        self._bridge_mgr = None

        self._setup_ttk_styles()
        self._build_header()
        self._build_notebook()
        self._build_status_bar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ttk_styles(self):
        s = ttk.Style(self)
        s.theme_use("default")

        s.configure(
            "TCombobox",
            fieldbackground=BG3,
            background=BG3,
            foreground=TEXT,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            borderwidth=0,
            arrowcolor=TEXT,
        )

        s.configure(
            "TProgressbar",
            troughcolor=BG3,
            background=ACCENT,
            thickness=12,
            borderwidth=0,
        )

        s.configure(
            "TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        s.configure(
            "TNotebook.Tab",
            background=TAB_BG_INACTIVE,
            foreground=TAB_FG_INACTIVE,
            padding=[18, 7],
            font=(FONT_FAMILY, 10, "bold"),
        )
        s.map(
            "TNotebook.Tab",
            background=[("selected", TAB_BG_ACTIVE),
                        ("active", "#3a3a70")],
            foreground=[("selected", TAB_FG_ACTIVE),
                        ("active", "#ffffff")],
        )

    def _build_header(self):
        hdr = tk.Frame(self, bg="#0d0d20", pady=0)
        hdr.pack(fill="x")

        inner = tk.Frame(hdr, bg="#0d0d20")
        inner.pack(fill="x", padx=20, pady=10)

        logo_f = tk.Frame(inner, bg="#0d0d20")
        logo_f.pack(side="left")
        tk.Label(
            logo_f, text="HPM", bg="#0d0d20", fg=ACCENT2,
            font=(FONT_FAMILY, 24, "bold"),
        ).pack(side="left")
        tk.Label(
            logo_f, text=" Psychophysiology System", bg="#0d0d20", fg=TEXT,
            font=(FONT_FAMILY, 12),
        ).pack(side="left", pady=4)

        tog = tk.Frame(inner, bg="#0d0d20")
        tog.pack(side="right")
        tk.Label(
            tog, text="Mode:", bg="#0d0d20", fg=TEXT,
            font=(FONT_FAMILY, 10, "bold"),
        ).pack(side="left", padx=(0, 8))

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

    def _build_notebook(self):
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True)

        # Scrollable containers for each tab
        self._tab_wizard_outer, self._tab_wizard = make_scrollable(self._nb)
        self._tab_monitor_outer, self._tab_monitor = make_scrollable(self._nb)
        self._tab_pavlovia_outer, self._tab_pavlovia = make_scrollable(self._nb)
        self._tab_log_outer, self._tab_log = make_scrollable(self._nb)
        self._tab_settings_outer, self._tab_settings = make_scrollable(self._nb)

        # Add outer frames (with scrollbars) to the notebook
        self._nb.add(self._tab_wizard_outer, text=" Setup Wizard ")
        self._nb.add(self._tab_monitor_outer, text=" Session Monitor ")
        self._nb.add(self._tab_pavlovia_outer, text=" Pavlovia URLs ")
        self._nb.add(self._tab_log_outer, text=" Log ")
        self._nb.add(self._tab_settings_outer, text=" Settings ")

        # Build content into the inner frames
        self._rebuild_wizard()
        self._build_monitor_tab()
        self._build_pavlovia_tab()
        self._build_log_tab()
        self._build_settings_tab()

    def _rebuild_wizard(self):
        for w in self._tab_wizard.winfo_children():
            w.destroy()
        mode = self._mode.get()
        steps = WIZARD_STEPS_RA if mode == "ra" else WIZARD_STEPS_PARTICIPANT
        SetupWizard(
            self._tab_wizard,
            steps=steps,
            mode=mode,
            on_complete=self._on_wizard_complete,
            subject_var=self._subject_var,
            session_var=self._session_var,
            version_var=self._version_var,
            app_ref=self,
        ).pack(fill="both", expand=True)

    def _on_mode_change(self):
        self._rebuild_wizard()

    def _on_wizard_complete(self):
        self._do_launch()

    def _build_monitor_tab(self):
        ctrl = tk.Frame(self._tab_monitor, bg=BG2, pady=8)
        ctrl.pack(fill="x", padx=14, pady=10)
        styled_button(
            ctrl, "▶ Launch Bridge",
            self._do_launch, style="success", width=18,
        ).pack(side="left", padx=6)
        styled_button(
            ctrl, "⏹ Stop Bridge",
            self._stop_bridge, style="danger", width=16,
        ).pack(side="left", padx=4)
        tk.Label(
            ctrl,
            text="(or complete the Setup Wizard to launch automatically)",
            bg=BG2, fg=TEXT2, font=(FONT_FAMILY, 9),
        ).pack(side="left", padx=10)

        self._monitor = SessionMonitor(self._tab_monitor, app_ref=self)
        self._monitor.pack(fill="both", expand=True)

    def _build_pavlovia_tab(self):
        self._pavlovia_panel = PavloviaPanel(
            self._tab_pavlovia,
            subject_var=self._subject_var,
        )
        self._pavlovia_panel.pack(fill="both", expand=True)

    def _build_log_tab(self):
        hdr = tk.Frame(self._tab_log, bg=BG, pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(
            hdr, text="Bridge Output Log", bg=BG, fg=TEXT,
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(side="left")
        styled_button(
            hdr, "Clear", self._clear_log,
            style="ghost", width=8,
        ).pack(side="right")

        self._log_text = tk.Text(
            self._tab_log, bg=BG3, fg=TEXT,
            font=("Courier", 10), relief="flat", bd=0,
            state="disabled", insertbackground=TEXT,
            selectbackground=ACCENT,
        )
        self._log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._log_text.tag_config("ok", foreground=SUCCESS)
        self._log_text.tag_config("warn", foreground=WARNING)
        self._log_text.tag_config("error", foreground=DANGER)
        self._log_text.tag_config("marker", foreground=ACCENT2)
        self._log_text.tag_config("default", foreground=TEXT)

    def _append_log(self, line):
        lo = line.lower()
        if any(w in lo for w in ("error", "fail", "✗", "traceback")):
            tag = "error"
        elif any(w in lo for w in ("warning", "⚠", "warn")):
            tag = "warn"
        elif any(w in lo for w in ("✓", "started", "saved", "complete",
                                   "connected", "▶", "⏹")):
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

    _MARKER_RE = re.compile(r"→ EVENT:\s*(.+)", re.IGNORECASE)
    _FPS_RE = re.compile(r"Recording FPS:\s*([\d.]+)", re.IGNORECASE)
    _FRAMES_RE = re.compile(r"Frames captured:\s*(\d+)", re.IGNORECASE)
    _TTL_RE_MON = re.compile(r"TTL PULSE CONFIRMED", re.IGNORECASE)

    def _route_to_monitor(self, line):
        m = self._FPS_RE.search(line)
        if m:
            self._monitor.update_stat("fps", f"{float(m.group(1)):.1f}")
        m = self._FRAMES_RE.search(line)
        if m:
            self._monitor.update_stat("frames", m.group(1))

        m = self._MARKER_RE.search(line)
        if m:
            marker_name = m.group(1).strip()
            ts_m = re.match(r"\[([\d:. ]+)\]", line)
            ts = ts_m.group(1).strip() if ts_m else _now()
            self._monitor.log_marker(ts, marker_name)

        if self._TTL_RE_MON.search(line):
            self._monitor.increment_ttl()

        if "Connected to Arduino on" in line:
            import re as _re
            _pm = _re.search(r"Connected to Arduino on\s+(\S+)", line)
            if _pm:
                self._monitor._port_var.set(_pm.group(1).strip())

        if "STARTING EXPERIMENT LOGGING" in line:
            self._monitor.update_stat("state", "RECORDING", SUCCESS)
        if "Physio log created:" in line:
            import re as _re
            _m = _re.search(r"Physio log created:\s*(.+\.csv)", line)
            if _m:
                _csv_path = _m.group(1).strip()
                if not os.path.isabs(_csv_path) and self._bridge_mgr:
                    _bridge_dir = os.path.dirname(
                        os.path.abspath(self._bridge_mgr.script_path)
                    )
                    _csv_path = os.path.join(_bridge_dir, _csv_path)
                self._monitor.set_csv_path(_csv_path)
                self._append_log(f"[{_now()}] CSV path: {_csv_path}")
        elif "EXPERIMENT END RECEIVED" in line:
            self._monitor.update_stat("state", "DONE", WARNING)
        elif "LAUNCHING POST-ACQUISITION ANALYSIS" in line:
            self._monitor.update_stat("state", "ANALYSING", ACCENT2)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _on_bridge_log(self, line):
        self.after(0, self._append_log, line)

    def _build_settings_tab(self):
        tk.Label(
            self._tab_settings, text="Settings", bg=BG, fg=TEXT,
            font=(FONT_FAMILY, 18, "bold"),
        ).pack(anchor="w", padx=24, pady=(18, 2))
        tk.Label(
            self._tab_settings,
            text="These values configure the bridge and pipeline scripts.",
            bg=BG, fg=TEXT2, font=(FONT_FAMILY, 10),
        ).pack(anchor="w", padx=24)
        tk.Frame(self._tab_settings, bg=BORDER, height=1).pack(
            fill="x", padx=24, pady=12
        )

        self._settings_vars = {}
        self._gsr_vcc_var = tk.StringVar(value="5.0")
        self._gsr_rref_var = tk.StringVar(value="100000")
        self._gsr_adcmax_var = tk.StringVar(value="32767")

        fields = [
            ("Participant ID", self._subject_var,
             "Shared with the Setup Wizard."),
            ("WebSocket Port", tk.StringVar(value=str(WEBSOCKET_PORT)),
             "Must match Pavlovia JS (default 5678)."),
            ("Target FPS", tk.StringVar(value="60"),
             "Video recording frame rate."),
            ("Calibration Duration (s)", tk.StringVar(value="30"),
             "Resting GSR baseline length."),
            ("Log Directory", tk.StringVar(value=LOG_DIR),
             "Folder for all session files."),
            ("GSR Vcc (V)", self._gsr_vcc_var,
             "CJMCU-6701 supply voltage (3.3 or 5.0)."),
            ("GSR Rref (Ω)", self._gsr_rref_var,
             "Reference resistor in ohms (default 100000)."),
            ("GSR ADC max", self._gsr_adcmax_var,
             "ADS1115=32767 (default), 12-bit=4095, 10-bit=1023."),
        ]

        for lbl_txt, var, hint in fields:
            row = tk.Frame(self._tab_settings, bg=BG)
            row.pack(fill="x", padx=24, pady=5)
            tk.Label(
                row, text=lbl_txt, bg=BG, fg=TEXT,
                font=(FONT_FAMILY, 11, "bold"),
                width=26, anchor="w",
            ).pack(side="left")
            tk.Entry(
                row, textvariable=var, font=(FONT_FAMILY, 11),
                bg=BG3, fg=TEXT, insertbackground=TEXT,
                relief="flat", bd=4, width=22,
            ).pack(side="left", padx=8)
            tk.Label(
                row, text=hint, bg=BG, fg=TEXT2,
                font=(FONT_FAMILY, 9),
            ).pack(side="left")
            self._settings_vars[lbl_txt] = var

        tk.Frame(self._tab_settings, bg=BORDER, height=1).pack(
            fill="x", padx=24, pady=14
        )
        tk.Label(
            self._tab_settings, text="Dependency Check", bg=BG, fg=TEXT,
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(anchor="w", padx=24)

        dep_frame = tk.Frame(self._tab_settings, bg=BG2, padx=12, pady=12)
        dep_frame.pack(fill="x", padx=24, pady=8)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        deps = [
            ("pyserial", SERIAL_AVAILABLE),
            ("numpy", NUMPY_AVAILABLE),
            ("opencv-python", CV2_AVAILABLE),
            (BRIDGE_SCRIPT, os.path.exists(os.path.join(base_dir, BRIDGE_SCRIPT))),
            (PIPELINE_SCRIPT, os.path.exists(os.path.join(base_dir, PIPELINE_SCRIPT))),
        ]

        for col, (name, ok) in enumerate(deps):
            f = tk.Frame(dep_frame, bg=BG2, padx=6)
            f.grid(row=0, column=col, padx=8)
            tk.Label(
                f, text="✓" if ok else "✗",
                fg=SUCCESS if ok else DANGER, bg=BG2,
                font=(FONT_FAMILY, 16, "bold"),
            ).pack()
            tk.Label(
                f, text=name, fg=TEXT if ok else DANGER, bg=BG2,
                font=(FONT_FAMILY, 8), wraplength=100,
                justify="center",
            ).pack()

    def _build_status_bar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._sbar = StatusBar(self)
        self._sbar.pack(fill="x")
        self._sbar.set("version", "HPM | Bridge v5.2.2 • Pipeline v7.17.2", TEXT2)
        self._sbar.set("bridge", "● Bridge: IDLE", TEXT2)
        self._sbar.set("clock", _now(), TEXT2)
        self._tick_clock()

    def _tick_clock(self):
        self._sbar.set("clock", _now())
        self.after(1000, self._tick_clock)

    def _do_launch(self):
        subject = self._subject_var.get().strip()
        if not subject:
            subject = f"subj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        bridge_path = os.path.join(base_dir, BRIDGE_SCRIPT)

        if not os.path.exists(bridge_path):
            messagebox.showerror(
                "Script Not Found",
                f"Could not find:\n{bridge_path}\n\n"
                f"Place {BRIDGE_SCRIPT} in the same folder as hpm_gui.py."
            )
            return

        if self._bridge_mgr and self._bridge_mgr.is_running():
            messagebox.showinfo(
                "Already Running",
                "The bridge is already running.\nStop it first to restart."
            )
            return

        self._bridge_mgr = BridgeManager(
            bridge_path,
            subject_id=subject,
            log_cb=self._on_bridge_log,
        )
        ok = self._bridge_mgr.start()
        if ok:
            # Update monitor state
            self._monitor._bridge_active = True
            self._monitor._stop_monitoring()
            self._monitor.set_canvas_mode(False)
            self._sbar.set("bridge", "● Bridge: RUNNING", SUCCESS)
            self._monitor.update_stat("state", "RUNNING", SUCCESS)
            self._monitor.start_timer()

            # Select the outer monitor tab (scrollable container)
            if hasattr(self, "_tab_monitor_outer"):
                self._nb.select(self._tab_monitor_outer)
            else:
                self._nb.select(self._tab_monitor)

            self._append_log(f"[{_now()}] Bridge started — subject: {subject}")
        else:
            messagebox.showerror(
                "Launch Failed",
                "Could not start the bridge script.\nCheck the Log tab for details."
            )

    def _stop_bridge(self):
        if self._bridge_mgr:
            self._bridge_mgr.stop()
            self._monitor._bridge_active = False
            self._monitor.set_canvas_mode(True)
            self._monitor._mon_status.config(
                text="Bridge stopped — click Start Monitoring for live signals",
                fg=TEXT2,
            )
            self._sbar.set("bridge", "● Bridge: STOPPED", DANGER)
            self._monitor.update_stat("state", "STOPPED", DANGER)
            self._monitor.stop_timer()
            self._append_log(f"[{_now()}] Bridge stopped.")

    def _on_close(self):
        if self._bridge_mgr and self._bridge_mgr.is_running():
            if not messagebox.askyesno(
                "Quit",
                "The bridge is still running.\nStop it and quit?"
            ):
                return
            self._bridge_mgr.stop()
        self.after(500, self.destroy())

def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    app = HPMApp()
    app.mainloop()

