#!/usr/bin/env python3
"""
PSYCHOPHYSIOLOGY PIPELINE v7.17
==============================================================
Multi-Method rPPG + Full Diagnostic Suite
--------------------------------------------------------------
NEW in v7.16:
1. GREEN-CHANNEL rPPG â€” baseline method
2. CHROM rPPG â€” chrominance-based (de Haan & Jeanne 2013)
3. POS rPPG â€” plane-orthogonal-to-skin (Wang et al. 2017)
4. ECG-ADAPTIVE rPPG â€” LMS adaptive filter, ECG as ground truth
5. Fig7 fully rebuilt â€” 4-method comparison panel
6. Fig10 rPPG diagnostics â€” Bland-Altman x3, correlation x3, SNR,
   raw signal traces, PSD comparison, temporal error, method ranking
7. Phase defaulting â€” full session = Habituation when no acq boundary
8. HR smoothing â€” rolling median + Gaussian
9. Column name aliases â€” ArduinoTimems / GSRuS / ECGmV all accepted
10. EDA DECONVOLUTION â€” replaces high-pass filter with biophysical sparse
    deconvolution (NeuroKit2 primary / Wiener+NNLS fallback). Eliminates
    filter-ringing sinusoidal artefact in phasic EDA. SCR peaks stored.
"""

import argparse, os, sys, warnings, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt, filtfilt, find_peaks, welch, spectrogram as stft
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import stats
import cv2

try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = np.trapz

warnings.filterwarnings('ignore')
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False
    print("WARNING: NeuroKit2 not installed.")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.titlesize': 16, 'figure.titleweight': 'bold',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5, 'text.usetex': False
})
sns.set_style("whitegrid")

STIM_COL = {'CS-': '#2ca02c', 'CS+': '#1f77b4', 'US': '#d62728'}
PHASE_COL = {'Habituation': '#4878CF', 'Acquisition': '#E87722'}
STIM_MAP = {'CSm.png': 'CS-', 'CSp.png': 'CS+', 'CSp_scream.png': 'US'}
METHOD_COL = {'Green': '#2196F3', 'CHROM': '#FF9800', 'POS': '#9C27B0', 'Adaptive': '#F44336'}

# ============================================================================
# HELPERS
# ============================================================================
def _canonical_scr_crf(sr):
    t = np.linspace(0, 30, int(30 * sr))
    h = (t / 2.0) ** 2 * np.exp(-t / 2.0)
    return h / np.max(h)

def _design_matrix(events, n, sr):
    if 'Phase' not in events.columns:
        events = events.copy(); events['Phase'] = 'Habituation'
    X, names = [np.ones(n), np.linspace(-1, 1, n)], ['Constant', 'LinearDrift']
    crf = _canonical_scr_crf(sr)
    def conv(onsets):
        s = np.zeros(n); v = onsets[(onsets >= 0) & (onsets < n)]; s[v] = 1.0
        return np.convolve(s, crf)[:n]
    for ph in ['Habituation', 'Acquisition']:
        for st in ['CS-', 'CS+']:
            mask = (events['Phase'] == ph) & (
                events['stimulus_type'].isin(['CS+', 'US']) if st == 'CS+' else
                events['stimulus_type'] == st)
            if mask.any():
                X.append(conv((events.loc[mask, 'time_sec'] * sr).astype(int).values))
                names.append(f'{ph}_{st}')
    us = events['stimulus_type'] == 'US'
    if us.any():
        X.append(conv(((events.loc[us, 'time_sec'] + 5.0) * sr).astype(int).values))
        names.append('US_Scream')
    return np.array(X).T, names

def _fit_glm(y, X):
    mask = np.isfinite(y)
    if mask.sum() < len(y) * 0.5:
        return np.zeros(X.shape[1]), np.zeros_like(y), y
    b, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return b, X @ b, y - X @ b

def _jitter(x, s=0.01): return x + np.random.normal(0, s, len(x))

def _caption(fig, text):
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.05, 0.02, textwrap.fill(text, 110), ha='left', va='bottom',
             fontsize=11, fontfamily='sans-serif', style='normal', color='black')

def _bland_altman(ax, ref, method, label, color):
    d = ref - method; m = (ref + method) / 2
    md, sd = d.mean(), d.std()
    ax.scatter(_jitter(m, 0.5), _jitter(d, 0.5), alpha=0.2, s=12, c=color, edgecolors='none')
    ax.axhline(md, c=color, lw=2, ls='-', label=f'Bias {md:.1f}')
    ax.axhline(md + 1.96*sd, c=color, lw=1.2, ls='--', label=f'+1.96SD {md+1.96*sd:.1f}')
    ax.axhline(md - 1.96*sd, c=color, lw=1.2, ls='--', label=f'-1.96SD {md-1.96*sd:.1f}')
    ax.axhline(0, c='k', lw=0.8, ls=':')
    ax.set_title(f'Bland-Altman: ECG vs {label}')
    ax.set_xlabel('Average BPM'); ax.set_ylabel('ECG \u2212 Method (BPM)')
    ax.legend(fontsize=8, frameon=True)
    return md, sd

def _corr_plot(ax, ref, method, label, color):
    r, p = stats.pearsonr(ref, method)
    m, b = np.polyfit(ref, method, 1)
    ax.scatter(_jitter(ref, 0.5), _jitter(method, 0.5), alpha=0.2, s=12, c=color, edgecolors='none')
    xl = np.array([ref.min(), ref.max()])
    ax.plot(xl, m*xl+b, c=color, lw=2, label=f'r={r:.3f}, p={p:.3f}')
    ax.plot(xl, xl, 'k--', lw=0.8)
    ax.set_title(f'ECG vs {label}'); ax.set_xlabel('ECG (BPM)'); ax.set_ylabel(f'{label} (BPM)')
    ax.legend(fontsize=8, frameon=True)
    return r, p

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    PHYSIO_CSV = None; MARKERS_CSV = None; VIDEO_PATH = None
    FRAME_TIMING_CSV = None; OUTPUT_DIR = 'analysis_output'
    def __setattr__(self, k, v):
        _a = {'PHYSIOCSV': 'PHYSIO_CSV', 'MARKERSCSV': 'MARKERS_CSV',
              'VIDEOPATH': 'VIDEO_PATH', 'FRAMETIMINGCSV': 'FRAME_TIMING_CSV', 'OUTPUTDIR': 'OUTPUT_DIR'}
        super().__setattr__(_a.get(k, k), v)

class UnifiedPhysioPipeline:
    def __init__(self, config=None): self.config = config or Config()
    def run(self):
        cfg = self.config
        if not cfg.PHYSIO_CSV or not cfg.MARKERS_CSV:
            raise ValueError("Config.PHYSIO_CSV and Config.MARKERS_CSV are required.")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(cfg.PHYSIO_CSV))[0]
        prefix = os.path.join(cfg.OUTPUT_DIR, base.replace('physio_data_', 'results_v7_16_'))
        PhysiologyPipelineV7(cfg.PHYSIO_CSV, cfg.MARKERS_CSV, prefix,
                             cfg.VIDEO_PATH, cfg.FRAME_TIMING_CSV).run()

# ============================================================================
# MAIN PIPELINE
# ============================================================================
class PhysiologyPipelineV7:
    def __init__(self, physio_csv, markers_csv, output_prefix, video_path, frame_timing_csv):
        self.physio_csv = physio_csv; self.markers_csv = markers_csv
        self.output_prefix = output_prefix; self.video_path = video_path
        self.frame_timing_csv = frame_timing_csv
        self.fs = 249.8        # default; overridden below from actual timestamps
        self.rppg_methods = {}
        self.rppg_available = False
        self.phase_boundary_sec = None
        self.glm_results_eda = None
        self.hrv_metrics = {}; self.lf_hf_stats = {}
        self.scr_peaks   = np.array([], dtype=int)  # populated by _eda_deconvolve

        print(f"Loading {physio_csv}...")
        self.df = pd.read_csv(physio_csv)
        # Strip any accidental whitespace from column headers (common with some CSV writers)
        self.df.columns = self.df.columns.str.strip()
        self.markers = pd.read_csv(markers_csv)

        # Coerce all signal columns to numeric — any row with non-numeric
        # content (e.g. TTL debug strings that slipped through the bridge
        # filter) becomes NaN and is dropped rather than crashing.
        _num_cols = ['ArduinoTime_ms', 'ArduinoTimems',
                     'RawGSR', 'GSR_uS', 'GSR_uS'.replace('_',''),
                     'RawECG', 'ECG_mV', 'ECG_mV'.replace('_','')]
        for _c in _num_cols:
            if _c in self.df.columns:
                self.df[_c] = pd.to_numeric(self.df[_c], errors='coerce')
        # Drop rows where the timestamp column is NaN (corrupted rows)
        _tc_candidates = ['ArduinoTime_ms', 'ArduinoTimems']
        _tc_present = [c for c in _tc_candidates if c in self.df.columns]
        if _tc_present:
            self.df.dropna(subset=_tc_present[:1], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            n_dropped = len(pd.read_csv(physio_csv)) - len(self.df)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} non-numeric rows from physio CSV")

        tc = next((c for c in ['ArduinoTime_ms', 'ArduinoTimems'] if c in self.df.columns), None)
        if tc:
            self.df['time_sec'] = (self.df[tc] - self.df[tc].iloc[0]) / 1000.0
            # Auto-detect actual sampling rate from median inter-sample interval
            _dt = self.df['time_sec'].diff().dropna()
            _dt = _dt[_dt > 0]
            if len(_dt) > 10:
                _fs_actual = 1.0 / float(_dt.median())
                if 10 < _fs_actual < 5000:   # sanity bounds
                    self.fs = _fs_actual
                    print(f"  Auto-detected sampling rate: {self.fs:.1f} Hz")
        else:
            self.df['time_sec'] = np.arange(len(self.df)) / self.fs

        try:
            t0 = pd.to_datetime(self.df['PythonSystemTime']).iloc[0]
            self.markers['timestamp_dt'] = pd.to_datetime(self.markers['timestamp'])
            self.markers['time_sec'] = (self.markers['timestamp_dt'] - t0).dt.total_seconds()
        except Exception:
            pass

        # If time_sec was not assigned (PythonSystemTime missing/unparseable), fall back to 0
        if 'time_sec' not in self.markers.columns:
            self.markers['time_sec'] = 0.0

        self.markers['stimulus_type'] = self.markers['message'].map(STIM_MAP)
        self.events = self.markers.dropna(subset=['stimulus_type']).copy()

    def _phase_shading(self, ax):
        if not self.phase_boundary_sec: return
        tmax = self.df['time_sec'].max() / 60; pb = self.phase_boundary_sec / 60
        ax.axvspan(0, pb, color=PHASE_COL['Habituation'], alpha=0.06)
        ax.axvspan(pb, tmax, color=PHASE_COL['Acquisition'], alpha=0.06)
        ax.axvline(pb, color='gray', ls='--', lw=1.2)

    def _has_phase(self):
        return 'Phase' in self.events.columns and len(self.events) > 0

    def process_signals(self):
        print("Processing signals...")
        # Case/underscore-insensitive column matching — handles ECG_mV, ECGmV, ecg_mv, RawECG, etc.
        _cols_norm = {c.strip().lower().replace('_', ''): c.strip() for c in self.df.columns}
        print(f"  Physio CSV columns: {list(self.df.columns)}")

        # Drop the old explicit list detection; replace with normalised lookup
        # ecgmv matches ECG_mV; rawecg matches RawECG; fallback: any col starting with 'ecg'
        ecg_col = next(
            (orig for norm, orig in _cols_norm.items()
             if norm == 'ecgmv' or (norm == 'rawecg' and 'ecgmv' not in _cols_norm) or (norm.startswith('ecg') and 'ecgmv' not in _cols_norm and 'rawecg' not in _cols_norm)),
            None)
        if ecg_col is None:
            print(f"  ⚠ No ECG column found.  Available: {list(self.df.columns)}")
            return
        print(f"  ECG column -> '{ecg_col}'")

        gsr_col = next(
            (orig for norm, orig in _cols_norm.items() if 'gsr' in norm),
            None)

        ecg_raw_vals = self.df[ecg_col].values.astype(float)
        # Fill NaN before processing
        ecg_raw_vals = pd.Series(ecg_raw_vals).ffill().bfill().fillna(0.0).values

        # Minimum samples NeuroKit needs: its smoothing kernel = int(fs * 2/3)
        _nk_min = max(int(self.fs * 2 / 3) + 10, 200)
        if len(ecg_raw_vals) < _nk_min:
            print(f"  ⚠ ECG signal too short ({len(ecg_raw_vals)} samples < {_nk_min} needed for NeuroKit2)")
            return

        if NK_AVAILABLE:
            ecg_clean = nk.ecg_clean(ecg_raw_vals, sampling_rate=self.fs)
        else:
            _sos_bp = butter(2, [0.5/(self.fs/2), min(40.0/(self.fs/2), 0.99)], 'band', output='sos')
            ecg_clean = sosfiltfilt(_sos_bp, ecg_raw_vals)
        self.df['ECG_clean'] = ecg_clean
        self.df['ECG_peaks'] = 0

        # Robust R-peak detection: try NeuroKit2, fall back to scipy
        rr_idx = np.array([], dtype=int)
        try:
            if not NK_AVAILABLE:
                raise RuntimeError("NeuroKit2 not installed")
            _, waves = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
            rr_idx = waves.get('ECG_R_Peaks', np.array([]))
            if rr_idx is None: rr_idx = np.array([], dtype=int)
        except Exception as _nk_err:
            print(f"  ⚠ NeuroKit2 R-peak detection failed ({_nk_err}); using scipy fallback")
            from scipy.signal import find_peaks as _fp
            # Band-pass 5-20 Hz, then find prominent positive peaks
            _sos = butter(2, [5.0 / (self.fs / 2), min(20.0 / (self.fs / 2), 0.99)],
                          'band', output='sos')
            _ecg_bp = sosfiltfilt(_sos, ecg_clean)
            _min_dist = max(1, int(self.fs * 0.4))   # 400 ms refractory
            _height   = np.percentile(_ecg_bp, 75)
            rr_idx, _ = _fp(_ecg_bp, distance=_min_dist, height=_height)
            rr_idx = rr_idx.astype(int)
            print(f"  scipy fallback found {len(rr_idx)} R-peaks")
        if len(rr_idx):
            self.df.loc[rr_idx, 'ECG_peaks'] = 1
            rr = np.diff(rr_idx) / self.fs
            self.df['HR'] = np.nan
            if len(rr_idx) > 1:
                self.df.loc[rr_idx[1:], 'HR'] = 60.0 / rr
            self.df['HR'] = self.df['HR'].interpolate()
            win = max(3, int(self.fs * 3))
            self.df['HR'] = self.df['HR'].rolling(window=win, center=True, min_periods=1).median()
            self.df['HR'] = gaussian_filter1d(
                self.df['HR'].ffill().bfill().values,
                sigma=self.fs * 0.5)
            self.r_peak_times_sec = self.df['time_sec'].values[rr_idx] if len(rr_idx) else np.array([])
            self.rr_intervals = rr

        if gsr_col:
            eda_raw = self.df[gsr_col].values.astype(float)
            # Fill NaN before filtering (non-numeric rows become NaN after coerce)
            _eda_s = pd.Series(eda_raw)
            eda_raw = _eda_s.ffill().bfill().fillna(0.0).values
            # Guard: skip EDA processing if signal too short or constant
            if len(eda_raw) < 500 or np.std(eda_raw) < 1e-9:
                print("  ⚠ EDA signal too short or constant — skipping EDA processing")
            else:
                # Low-pass only: remove HF noise, preserve slow EDA dynamics
                sos_lp = butter(4, 3.0, 'lp', fs=self.fs, output='sos')
                eda_clean = sosfiltfilt(sos_lp, eda_raw)
                self.df['EDA_clean'] = eda_clean
                # â”€â”€ Biophysical deconvolution (v7.17) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Replaces high-pass filter which caused sinusoidal ringing.
                # _eda_deconvolve() uses NK2 sparse deconvolution (primary)
                # or Wiener+NNLS fallback â€” both are non-negative and
                # produce physiologically valid phasic waveforms.
                phasic, tonic, scr_peaks = self._eda_deconvolve(eda_clean)
                self.df['EDA_phasic'] = phasic
                self.df['EDA_tonic']  = tonic
                self.scr_peaks        = scr_peaks  # sample indices


    # -----------------------------------------------------------------------
    # EDA DECONVOLUTION (v7.17)
    # -----------------------------------------------------------------------
    def _eda_deconvolve(self, eda_clean):
        """
        Decompose EDA into tonic and phasic (driver) components using sparse
        biophysical deconvolution.

        Primary  : NeuroKit2 nk.eda_process()  (sparse deconvolution, Greco 2016)
        Fallback : Wiener deconvolution + NNLS non-negativity projection
                       (avoids the filter-ringing sinusoidal artefact produced by
                        naive high-pass filtering of the slow tonic baseline)

        Returns
        -------
        phasic    : ndarray  â€“ phasic EDA reconstructed from sparse driver
        tonic     : ndarray  â€“ slow tonic component
        scr_peaks : ndarray  â€“ sample indices of detected SCR onsets
        """
        n = len(eda_clean)

        # â”€â”€ Primary: NeuroKit2 sparse deconvolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if NK_AVAILABLE:
                try:
                    signals, info = nk.eda_process(eda_clean,
                                                   sampling_rate=int(self.fs))
                    phasic    = signals['EDA_Phasic'].values
                    tonic     = signals['EDA_Tonic'].values
                    scr_peaks = np.array(info.get('SCR_Peaks', []), dtype=int)
                    # Guard: NK sometimes returns float indices
                    scr_peaks = scr_peaks[scr_peaks < n].astype(int)
                    print(f"   EDA deconvolution (NeuroKit2 sparse): "
                          f"{len(scr_peaks)} SCR peaks detected")
                    return phasic, tonic, scr_peaks
                except Exception as e:
                    print(f"   NeuroKit2 EDA deconvolution failed ({e}); "
                          f"using Wiener fallback")

        # â”€â”€ Fallback: Wiener deconvolution + NNLS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Downsample to 8 Hz (EDA has no useful content above ~2 Hz)
        ds = max(1, int(self.fs / 8.0))
        fs8 = self.fs / ds
        y   = eda_clean[::ds]
        m   = len(y)

        # Tonic: very slow Butterworth LP (0.05 Hz)
        sos_t = butter(2, 0.05 / (fs8 / 2.0), 'lp', output='sos')
        tonic_ds = sosfiltfilt(sos_t, y)

        # Phasic residual
        res = y - tonic_ds

        # SCR kernel: two-component Bateman function (Ï„1=0.75s, Ï„2=2s)
        t_k = np.arange(0, 30.0, 1.0 / fs8)
        tau1, tau2 = 0.75, 2.0
        kernel = np.exp(-t_k / tau2) - np.exp(-t_k / tau1)
        kernel[kernel < 0] = 0.0
        peak_k = kernel.max()
        if peak_k > 0:
                kernel /= peak_k
        kn = len(kernel)

        # Wiener deconvolution in frequency domain
        nfft = int(2 ** np.ceil(np.log2(m + kn)))
        R  = np.fft.rfft(res,  n=nfft)
        H  = np.fft.rfft(kernel, n=nfft)
        H_pow  = np.abs(H) ** 2
        # Regularisation: 1% of peak power â†’ prevents noise amplification
        lam = 0.01 * H_pow.max()
        driver_raw = np.fft.irfft(R * np.conj(H) / (H_pow + lam), n=nfft)[:m]

        # Non-negativity projection (SCRs are strictly positive deflections)
        driver = np.maximum(driver_raw, 0.0)

        # Reconstruct phasic by convolving sparse driver with SCR kernel
        phasic_ds = np.convolve(driver, kernel, mode='full')[:m]

        # SCR peak detection on driver signal
        min_dist = max(1, int(fs8 * 4.0))   # 4-second refractory period
        threshold = np.percentile(driver[driver > 0], 60) if driver.max() > 0 else 0
        peak_ds, _ = find_peaks(driver, height=threshold, distance=min_dist)

        # Upsample back to original sampling rate
        t_orig = np.arange(n)
        t_ds   = np.arange(m) * ds
        phasic = np.interp(t_orig, t_ds, phasic_ds)
        tonic  = np.interp(t_orig, t_ds, tonic_ds)
        scr_peaks = (peak_ds * ds).astype(int)
        scr_peaks = scr_peaks[scr_peaks < n]

        print(f"   EDA deconvolution (Wiener+NNLS fallback): "
                  f"{len(scr_peaks)} SCR peaks detected")
        return phasic, tonic, scr_peaks

    def detect_phases(self):
        scream = self.markers[self.markers['message'] == 'CSp_scream.png']
        if len(scream):
            self.phase_boundary_sec = float(scream.iloc[0]['time_sec'])
        else:
            acq = self.markers[self.markers['message'].str.startswith('acq', na=False)]
            self.phase_boundary_sec = float(acq.iloc[0]['time_sec']) if len(acq) else None

        if self.phase_boundary_sec:
            self.df['Phase'] = np.where(self.df['time_sec'] < self.phase_boundary_sec,
                                        'Habituation', 'Acquisition')
            self.events['Phase'] = np.where(self.events['time_sec'] < self.phase_boundary_sec,
                                            'Habituation', 'Acquisition')
        else:
            print("  \u26a0 No acquisition boundary detected \u2014 labelling session as Habituation (pilot mode).")
            self.df['Phase'] = 'Habituation'; self.events['Phase'] = 'Habituation'
            self.phase_boundary_sec = self.df['time_sec'].max()

    def run_glm_analysis(self):
        if 'EDA_phasic' not in self.df.columns: return
        print("Running GLM...")
        X, names = _design_matrix(self.events, len(self.df), self.fs)
        b, pred, resid = _fit_glm(self.df['EDA_phasic'].values, X)
        self.glm_results_eda = {'betas': dict(zip(names, b)), 'pred': pred, 'resid': resid, 'X': X}

    def compute_metrics(self):
        if not hasattr(self, 'rr_intervals') or len(self.rr_intervals) == 0: return
        for phase in ['Habituation', 'Acquisition']:
            if self.phase_boundary_sec is None: continue
            try:
                m = (self.r_peak_times_sec[1:] < self.phase_boundary_sec if phase == 'Habituation'
                     else self.r_peak_times_sec[1:] >= self.phase_boundary_sec)
                rr = self.rr_intervals[m]
                if len(rr) > 4:
                    self.hrv_metrics[phase] = {'RMSSD': np.sqrt(np.mean(np.diff(rr * 1000) ** 2))}
                    t_rr = np.cumsum(rr); t_rr -= t_rr[0]
                    fi = interp1d(t_rr, rr, kind='cubic', fill_value='extrapolate')
                    ti = np.arange(0, t_rr[-1], 0.25); ri = fi(ti)
                    nperseg = min(len(ri), 256)
                    if nperseg >= 32:
                        f, p = welch(ri, fs=4.0, nperseg=nperseg)
                        lf = trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
                        hf = trapezoid(p[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
                        self.lf_hf_stats[phase] = {'LF': lf, 'HF': hf, 'Ratio': lf / hf if hf > 0 else np.nan}
            except Exception as e:
                print(f"  Spectral error {phase}: {e}")

    # -----------------------------------------------------------------------
    # MULTI-METHOD rPPG
    # -----------------------------------------------------------------------
    def process_rppg(self):
        if not self.video_path or not self.frame_timing_csv: return
        print("Extracting rPPG (Green / CHROM / POS / Adaptive)...")
        try:
            ft = pd.read_csv(self.frame_timing_csv)
            ec = 'elapsed_sec' if 'elapsed_sec' in ft.columns else 'elapsedsec'
            fc = 'frame_num' if 'frame_num' in ft.columns else 'framenum'
            frame_time = ft[ec].values; frame_num = ft[fc].values

            exp_starts = self.markers[self.markers['message'] == 'experiment_start']
            if len(exp_starts) >= 1:
                exp_ev = exp_starts.iloc[0]
                t_physio_ev = float(exp_ev['time_sec']) if 'time_sec' in exp_ev.index else 0.0
                ev_frame = int(exp_ev['frame_number'])
                i_ev = np.argmin(np.abs(frame_num - ev_frame))
                t_video_ev = float(frame_time[i_ev])
                time_offset = t_physio_ev - t_video_ev
            else:
                time_offset = 0.0

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened(): return
            n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frame_time))
            R, G, B = [], [], []
            # rPPG needs ~30 Hz max — skip every other frame at 60 fps (2x speedup)
            # and resize ROI to 320x240 before averaging (4-8x speedup vs 1080p crop)
            _stride = max(1, round(rppg_fs_hint / 30.0)) if (rppg_fs_hint := (
                1.0 / np.median(np.diff(frame_time[:min(120, len(frame_time))])) if len(frame_time) > 2 else 60.0
            )) > 30 else 1
            _frames_to_read = list(range(0, n_frames, _stride))
            _n_use = len(_frames_to_read)
            print(f"  Reading {_n_use}/{n_frames} video frames (stride={_stride}, target ~30 Hz)...", flush=True)
            _prog_step = max(1, _n_use // 20)
            _frame_idx = 0  # actual frame counter in video
            _use_idx   = 0  # index into _frames_to_read
            while _use_idx < _n_use:
                target = _frames_to_read[_use_idx]
                # Seek only if we need to jump (seek is expensive; avoid when stride==1)
                if _stride > 1 and _frame_idx != target:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    _frame_idx = target
                ret, frame = cap.read()
                if not ret:
                    break
                _frame_idx += 1
                h, w = frame.shape[:2]
                # Resize to at most 320 wide before averaging — same result, much faster
                _scale = min(1.0, 320.0 / w)
                if _scale < 1.0:
                    frame = cv2.resize(frame, (int(w * _scale), int(h * _scale)),
                                       interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                roi = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
                B.append(float(roi[:, :, 0].mean()))
                G.append(float(roi[:, :, 1].mean()))
                R.append(float(roi[:, :, 2].mean()))
                _use_idx += 1
                if _use_idx % _prog_step == 0 or _use_idx == _n_use:
                    pct = int(100 * _use_idx / _n_use)
                    print(f"  rPPG frame extraction: {_use_idx}/{_n_use} ({pct}%)", flush=True)
            cap.release()
            # Re-align frame_time to the strided indices
            frame_time = frame_time[_frames_to_read[:len(G)]]

            R = np.array(R); G = np.array(G); B = np.array(B)
            ft_use = frame_time[:len(G)]
            rppg_fs = 1.0 / np.median(np.diff(ft_use))
            print(f"  Video FPS estimated: {rppg_fs:.1f}")

            def _extract_bpm(signal, name):
                b, a = butter(2, [0.7 / (rppg_fs / 2), 2.5 / (rppg_fs / 2)], 'band')
                sig = filtfilt(b, a, signal - signal.mean())
                win_s = int(10 * rppg_fs); step_s = int(1.0 * rppg_fs)
                bpms, btimes = [], []
                for s in range(0, len(sig) - win_s, step_s):
                    seg = sig[s:s + win_s]
                    pks, _ = find_peaks(seg, distance=int(rppg_fs * 0.4))
                    val = np.median(60.0 / (np.diff(pks) / rppg_fs)) if len(pks) >= 2 else np.nan
                    bpms.append(val); btimes.append(ft_use[s:s + win_s].mean())
                bpms = np.array(bpms)
                aligned = np.array(btimes) + time_offset
                interped = np.interp(self.df['time_sec'].values, aligned, bpms,
                                     left=np.nan, right=np.nan)
                self.df[f'HR_{name}'] = interped
                self.rppg_methods[name] = interped
                print(f"  {name} rPPG: mean {np.nanmean(interped):.1f} BPM")
                return sig

            # Method 1: Green channel
            green_sig = _extract_bpm(G, 'Green')

            # Method 2: CHROM (de Haan & Jeanne 2013)
            Rn = R / (R.mean() + 1e-9); Gn = G / (G.mean() + 1e-9); Bn = B / (B.mean() + 1e-9)
            Xs = 3 * Rn - 2 * Gn; Ys = 1.5 * Rn + Gn - 1.5 * Bn
            alpha = np.std(Xs) / (np.std(Ys) + 1e-9)
            chrom_sig = Xs - alpha * Ys
            _extract_bpm(chrom_sig, 'CHROM')

            # Method 3: POS (Wang et al. 2017)
            C = np.stack([R, G, B], axis=1)
            Cn = C / (C.mean(axis=0, keepdims=True) + 1e-9)
            P = np.array([[0, 1, -1], [-2, 1, 1]])
            S = P @ Cn.T
            s1, s2 = S[0], S[1]
            pos_sig = s1 + (np.std(s1) / (np.std(s2) + 1e-9)) * s2
            _extract_bpm(pos_sig, 'POS')

            # Method 4: ECG-Adaptive (LMS filter)
            if 'HR' in self.df.columns:
                ecg_hr = self.df['HR'].values
                chrom_hr = self.df['HR_CHROM'].values
                train_n = int(30 * self.fs)
                valid = np.isfinite(ecg_hr[:train_n]) & np.isfinite(chrom_hr[:train_n])
                if valid.sum() > 100:
                    X_tr = np.stack([np.ones(valid.sum()), chrom_hr[:train_n][valid]], axis=1)
                    y_tr = ecg_hr[:train_n][valid]
                    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
                    adapted = w[0] + w[1] * chrom_hr
                    adapted[~np.isfinite(adapted)] = np.nan
                    self.df['HR_Adaptive'] = adapted
                    self.rppg_methods['Adaptive'] = adapted
                    print(f"  Adaptive rPPG: w=[{w[0]:.2f}, {w[1]:.3f}], mean {np.nanmean(adapted):.1f} BPM")

            self._rppg_raw = {'Green': green_sig, 'CHROM': chrom_sig, 'POS': pos_sig}
            self._rppg_ft  = ft_use + time_offset  # physio-aligned time axis
            self._rppg_fs  = rppg_fs
            self.rppg_available = len(self.rppg_methods) > 0

        except Exception as e:
            import traceback
            print(f"rPPG failed: {e}"); traceback.print_exc()

    # -----------------------------------------------------------------------
    # FIGURES
    # -----------------------------------------------------------------------
    def plot_fig1_overview(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        t = self.df['time_sec'].values / 60
        if 'ECG_clean' in self.df.columns:
            axes[0].plot(t, self.df['ECG_clean'], 'k-', lw=0.35)
            axes[0].set_title('A) ECG (Raw)'); axes[0].set_ylabel('Amplitude (mV)')
            self._phase_shading(axes[0])
        if 'HR' in self.df.columns:
            axes[1].plot(t, self.df['HR'], 'b-', lw=1.5, label='ECG-HR')
            for name, col in METHOD_COL.items():
                if f'HR_{name}' in self.df.columns:
                    axes[1].plot(t, self.df[f'HR_{name}'], '-', color=col, lw=0.9, alpha=0.7, label=name)
            axes[1].legend(frameon=True, fontsize=9); axes[1].set_title('B) Heart Rate (BPM)')
            axes[1].set_ylabel('BPM'); self._phase_shading(axes[1])
        if 'EDA_clean' in self.df.columns:
            axes[2].plot(t, self.df['EDA_tonic'],  'k-', lw=0.5, alpha=0.6, label='Tonic')
            axes[2].plot(t, self.df['EDA_phasic'], 'r-', lw=1.2, label='Phasic (deconvolved)')
            if len(self.scr_peaks):
                scr_t = self.df['time_sec'].values[self.scr_peaks] / 60
                scr_y = self.df['EDA_phasic'].values[self.scr_peaks]
                axes[2].plot(scr_t, scr_y, 'rv', ms=7, label='SCR peaks', zorder=5)
            axes[2].legend(frameon=True)
            axes[2].set_title('C) EDA (biophysical deconvolution)'); axes[2].set_ylabel('Conductance (uS)')
            self._phase_shading(axes[2]); axes[2].set_xlabel('Time (min)')
        _caption(fig, "Figure 1. Signal Overview. (A) ECG. (B) HR from ECG (blue) and all rPPG methods. (C) EDA tonic and phasic components. Shaded: Habituation (blue), Acquisition (orange).")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig1_Overview.png', dpi=300); plt.close()

    def plot_fig2_hrv(self):
        if not hasattr(self, 'rr_intervals') or len(self.rr_intervals) == 0: return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = self.r_peak_times_sec[1:] / 60
        rr_ms = self.rr_intervals * 1000
        axes[0, 0].plot(t, rr_ms, 'k-', lw=0.8); axes[0, 0].set_title('A) RR Tachogram')
        axes[0, 0].set_ylabel('RR (ms)'); axes[0, 0].set_xlabel('Time (min)')
        self._phase_shading(axes[0, 0])
        if len(rr_ms) > 4:
            axes[0, 1].scatter(_jitter(rr_ms[:-1], 1.5), _jitter(rr_ms[1:], 1.5),
                               alpha=0.4, s=15, c='steelblue', edgecolors='none')
            mn, mx = rr_ms.min(), rr_ms.max()
            axes[0, 1].plot([mn, mx], [mn, mx], 'k--', lw=0.8)
            axes[0, 1].set_title('B) Poincar\xe9'); axes[0, 1].set_xlabel('RR[n]'); axes[0, 1].set_ylabel('RR[n+1]')
        phases = ['Habituation', 'Acquisition']
        if all(p in self.hrv_metrics for p in phases):
            axes[1, 0].bar(phases, [self.hrv_metrics[p]['RMSSD'] for p in phases],
                           color=[PHASE_COL[p] for p in phases])
            axes[1, 0].set_title('C) RMSSD by Phase'); axes[1, 0].set_ylabel('RMSSD (ms)')
        axes[1, 1].axis('off')
        _caption(fig, "Figure 2. HRV. (A) RR tachogram. (B) Poincar\xe9 plot. (C) RMSSD by phase.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig2_HRV.png', dpi=300); plt.close()

    def plot_fig3_eda_glm(self):
        if not self.glm_results_eda or 'EDA_phasic' not in self.df.columns: return
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        t = self.df['time_sec'].values / 60
        axes[0, 0].plot(t, self.df['EDA_tonic'], 'steelblue')
        axes[0, 0].plot(t, self.df['EDA_phasic'], 'tomato')
        axes[0, 0].set_title('A) EDA Decomposition (biophysical deconvolution)'); self._phase_shading(axes[0, 0])
        axes[0, 0].set_xlabel('Time (min)'); axes[0, 0].set_ylabel('Conductance (uS)')
        axes[0, 1].plot(t, self.df['EDA_phasic'], 'k-', alpha=0.2, label='Deconvolved Phasic')
        axes[0, 1].plot(t, self.glm_results_eda['pred'], 'r-', lw=1.0, label='GLM Fit')
        if len(self.scr_peaks):
            scr_t2 = self.df['time_sec'].values[self.scr_peaks] / 60
            scr_y2 = self.df['EDA_phasic'].values[self.scr_peaks]
            axes[0, 1].plot(scr_t2, scr_y2, 'rv', ms=6, label='SCR peaks', zorder=5)
        axes[0, 1].legend(frameon=True); axes[0, 1].set_title('B) GLM Fit on Deconvolved Phasic')
        axes[0, 1].set_xlabel('Time (min)'); axes[0, 1].set_ylabel('Phasic EDA (uS)')
        betas = self.glm_results_eda['betas']
        keys = sorted([k for k in betas if 'Constant' not in k and 'Drift' not in k])
        axes[1, 0].barh(keys, [betas[k] for k in keys], color='teal')
        axes[1, 0].set_title('C) Beta Weights'); axes[1, 0].set_xlabel('Beta (uS)')
        axes[1, 1].plot(t, self.glm_results_eda['resid'], 'gray', lw=0.5)
        axes[1, 1].set_title('D) Residuals'); axes[1, 1].set_xlabel('Time (min)')
        crf = _canonical_scr_crf(self.fs); t_crf = np.arange(len(crf)) / self.fs
        for k in keys:
            if 'US' not in k: axes[2, 0].plot(t_crf, crf * betas[k], label=k, lw=2)
        if 'US_Scream' in betas: axes[2, 0].plot(t_crf + 5, crf * betas['US_Scream'], 'k--', label='US')
        axes[2, 0].legend(frameon=True); axes[2, 0].set_title('E) Response Shapes')
        axes[2, 0].set_xlabel('Time post-stimulus (s)'); axes[2, 0].set_ylabel('Response (uS)')
        axes[2, 1].axis('off')
        _caption(fig, "Figure 3. EDA GLM. (A) Tonic/phasic via sparse deconvolution (not high-pass filter). (B) GLM fit. (C) Beta weights. (D) Residuals. (E) Modeled response shapes.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig3_EDA_GLM.png', dpi=300); plt.close()

    def plot_fig4_spectral(self):
        if 'ECG_clean' not in self.df.columns: return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        f, p = welch(self.df['ECG_clean'], fs=self.fs, nperseg=int(4 * self.fs))
        axes[0, 0].semilogy(f, p, 'k', lw=1)
        axes[0, 0].fill_between(f, p, where=(f >= 0.04) & (f < 0.15), color='green', alpha=0.2, label='LF')
        axes[0, 0].fill_between(f, p, where=(f >= 0.15) & (f < 0.40), color='blue', alpha=0.2, label='HF')
        axes[0, 0].set_title('A) ECG PSD'); axes[0, 0].set_xlim(0, 0.6)
        axes[0, 0].set_xlabel('Frequency (Hz)'); axes[0, 0].legend(frameon=True)
        if 'EDA_clean' in self.df.columns:
            f, p = welch(self.df['EDA_clean'], fs=self.fs, nperseg=int(30 * self.fs))
            axes[0, 1].semilogy(f, p); axes[0, 1].set_title('B) EDA PSD')
            axes[0, 1].set_xlim(0, 2); axes[0, 1].set_xlabel('Frequency (Hz)')
            f, t_s, Sxx = stft(self.df['EDA_phasic'], fs=self.fs, nperseg=int(30 * self.fs))
            axes[1, 0].pcolormesh(t_s / 60, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
            axes[1, 0].set_title('C) EDA Spectrogram'); axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_xlabel('Time (min)'); axes[1, 0].set_ylabel('Frequency (Hz)')
        if any(p in self.lf_hf_stats for p in ['Habituation', 'Acquisition']):
            phases = [p for p in ['Habituation', 'Acquisition'] if p in self.lf_hf_stats]
            axes[1, 1].bar(phases, [self.lf_hf_stats[p]['Ratio'] for p in phases],
                           color=[PHASE_COL[p] for p in phases])
            axes[1, 1].set_title('D) LF/HF Ratio'); axes[1, 1].set_ylabel('Ratio')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            axes[1, 1].axis('off')
        _caption(fig, "Figure 4. Spectral QC. (A) ECG PSD with LF/HF bands. (B) EDA PSD. (C) EDA spectrogram. (D) LF/HF ratio.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig4_Spectral.png', dpi=300); plt.close()

    def plot_fig5_snr(self):
        if 'ECG_clean' not in self.df.columns: return
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        phases_present = self.df['Phase'].unique().tolist() if 'Phase' in self.df.columns else []
        for ax, col, title in [(axes[0], 'ECG_clean', 'A) ECG Distribution'),
                                (axes[1], 'EDA_tonic', 'B) EDA Distribution')]:
            if col not in self.df.columns: continue
            if phases_present:
                for ph in phases_present:
                    d = self.df[self.df['Phase'] == ph][col]
                    d_valid = d.dropna()
                    d_valid = d_valid[np.isfinite(d_valid)]
                    if len(d_valid) > 1:
                        ax.hist(d_valid, bins=50, alpha=0.5, label=ph, density=True)
                ax.legend(frameon=True)
            else:
                _d = self.df[col].dropna()
                _d = _d[np.isfinite(_d)]
                if len(_d) > 1:
                    ax.hist(_d, bins=50, alpha=0.7, density=True)
            ax.set_title(title); ax.set_ylabel('Density')
        _caption(fig, "Figure 5. Signal distributions by phase.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig5_SNR.png', dpi=300); plt.close()

    def plot_fig6_heatmaps(self):
        # Run even without phase labels — just use all available CS events
        if 'HR' not in self.df.columns: return
        if len(self.events) == 0: print("Skipping Fig6: no stimulus events."); return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        def _hm(ax, phase, col, title):
            if phase and self._has_phase():
                evs = self.events[(self.events['Phase'] == phase) &
                                  (self.events['stimulus_type'].isin(['CS+', 'CS-', 'US']))]
            else:
                evs = self.events[self.events['stimulus_type'].isin(['CS+', 'CS-', 'US'])]
            if len(evs) == 0: ax.axis('off'); return
            dat = []
            for et in evs['time_sec']:
                i = int(et * self.fs)
                if i + int(10 * self.fs) < len(self.df) and i - int(2 * self.fs) >= 0:
                    seg = self.df[col].values[i - int(2 * self.fs):i + int(10 * self.fs)]
                    dat.append(seg - np.mean(seg[:int(2 * self.fs)]))
            if dat:
                im = ax.imshow(dat, aspect='auto', cmap='RdBu_r', extent=[-2, 10, len(dat), 0])
                plt.colorbar(im, ax=ax); ax.axvline(0, color='w'); ax.axvline(5, color='y', ls='--')
                ax.set_title(title); ax.set_xlabel('Time (s)'); ax.set_ylabel('Trial')
        if self._has_phase():
            _hm(axes[0, 0], 'Habituation', 'HR', 'Hab: HR')
            _hm(axes[1, 0], 'Acquisition', 'HR', 'Acq: HR')
            _hm(axes[0, 1], 'Habituation', 'EDA_phasic', 'Hab: EDA')
            _hm(axes[1, 1], 'Acquisition', 'EDA_phasic', 'Acq: EDA')
        else:
            _hm(axes[0, 0], None, 'HR', 'All Trials: HR')
            axes[1, 0].axis('off')
            _hm(axes[0, 1], None, 'EDA_phasic', 'All Trials: EDA')
            axes[1, 1].axis('off')
        _caption(fig, "Figure 6. Single-trial heatmaps (CS+/US). Rows = trials, x = time from stimulus onset.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig6_Heatmaps.png', dpi=300); plt.close()

    def plot_fig7_rppg_comparison(self):
        if not self.rppg_available or 'HR' not in self.df.columns: return
        methods = [m for m in ['Green', 'CHROM', 'POS', 'Adaptive'] if f'HR_{m}' in self.df.columns]
        if not methods: return
        n = len(methods)
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
        if n == 1: axes = axes[np.newaxis, :]
        ecg_full = self.df['HR'].values
        t = self.df['time_sec'].values / 60
        summary_rows = []
        for row, m in enumerate(methods):
            mhr = self.df[f'HR_{m}'].values
            col = METHOD_COL.get(m, 'gray')
            valid = np.isfinite(ecg_full) & np.isfinite(mhr)
            if valid.sum() < 20:
                for c in range(3): axes[row, c].axis('off')
                continue
            ref = ecg_full[valid]; est = mhr[valid]
            r, p = stats.pearsonr(ref, est)
            rmse = np.sqrt(np.mean((ref - est) ** 2))
            axes[row, 0].plot(t, ecg_full, 'b-', lw=1.2, alpha=0.8, label='ECG')
            axes[row, 0].plot(t, mhr, '-', color=col, lw=1.0, alpha=0.8, label=m)
            axes[row, 0].set_title(f'{m} \u2014 Time Series'); axes[row, 0].set_ylabel('BPM')
            axes[row, 0].set_xlabel('Time (min)'); axes[row, 0].legend(fontsize=8, frameon=True)
            self._phase_shading(axes[row, 0])
            _bland_altman(axes[row, 1], ref, est, m, col)
            _corr_plot(axes[row, 2], ref, est, m, col)
            axes[row, 2].set_title(f'{m}: r={r:.3f}, RMSE={rmse:.1f} BPM')
            summary_rows.append({'Method': m, 'r': round(r, 3), 'RMSE_BPM': round(rmse, 1), 'p': round(p, 4)})
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(f'{self.output_prefix}_rPPG_summary.csv', index=False)
        _caption(fig, "Figure 7. Multi-Method rPPG vs ECG Comparison. For each method: (Left) HR time series overlay, (Centre) Bland-Altman agreement, (Right) Pearson correlation with line of best fit.")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig7_rPPG_Comparison.png', dpi=300); plt.close()

    def plot_fig8_zooms(self):
        if 'ECG_clean' not in self.df.columns: return
        # Auto-select a 10s window: prefer a window containing R-peaks,
        # otherwise use the session midpoint.
        t_all = self.df['time_sec'].values
        t_mid = t_all[len(t_all) // 2]
        if hasattr(self, 'r_peak_times_sec') and len(self.r_peak_times_sec) > 5:
            # Centre on the median R-peak time
            t_mid = float(np.median(self.r_peak_times_sec))
        t_start = max(t_all[0], t_mid - 5)
        t_end   = t_start + 10
        mask = (self.df['time_sec'] >= t_start) & (self.df['time_sec'] <= t_end)
        if not mask.any(): return
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        t = self.df.loc[mask, 'time_sec']
        axes[0].plot(t, self.df.loc[mask, 'ECG_clean'], 'k-', lw=1.5)
        if 'ECG_peaks' in self.df.columns:
            pks = self.df.loc[mask, 'ECG_peaks'] == 1
            axes[0].plot(t[pks], self.df.loc[mask, 'ECG_clean'][pks], 'ro', ms=5)
        axes[0].set_title('A) ECG 10s Zoom'); axes[0].set_ylabel('Amplitude (mV)')
        if 'EDA_clean' in self.df.columns:
            axes[1].plot(t, self.df.loc[mask, 'EDA_clean'], 'k-', lw=1.5, label='Raw')
            axes[1].plot(t, self.df.loc[mask, 'EDA_phasic'] + self.df.loc[mask, 'EDA_clean'].mean(),
                         'r-', lw=1.0, label='Phasic')
            axes[1].legend(frameon=True)
            axes[1].set_title('B) EDA 10s Zoom'); axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Conductance (uS)')
        _caption(fig, "Figure 8. Signal Quality Zoom. (A) 10s ECG with R-peaks. (B) 10s EDA raw vs phasic.")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig8_SignalZoom.png', dpi=300); plt.close()

    def plot_fig9_averages(self):
        if 'HR' not in self.df.columns: return
        if len(self.events) == 0: print("Skipping Fig9: no stimulus events."); return
        # Use phase labels if available, otherwise treat entire session as one group
        phases_present = (sorted(self.events['Phase'].unique().tolist())
                          if self._has_phase() else ['All Trials'])
        n_phases = len(phases_present)
        fig, axes = plt.subplots(2, n_phases, figsize=(7 * n_phases, 10), squeeze=False)
        signals = [s for s in [('HR', 'Heart Rate (BPM)'),
                                ('EDA_phasic', 'Phasic EDA (uS)')]
                   if s[0] in self.df.columns]
        for i, (sig, ylab) in enumerate(signals):
            for j, phase in enumerate(phases_present):
                ax = axes[i, j]
                for stim_type, color in STIM_COL.items():
                    if stim_type == 'US' and phase == 'Habituation': continue
                    if phase == 'All Trials':
                        evs = self.events[self.events['stimulus_type'] == stim_type]
                    else:
                        evs = self.events[(self.events['Phase'] == phase) &
                                          (self.events['stimulus_type'] == stim_type)]
                    if len(evs) == 0: continue
                    segs = []
                    for et in evs['time_sec']:
                        idx = int(et * self.fs)
                        pre = int(2 * self.fs); post = int(8 * self.fs)
                        if idx + post < len(self.df) and idx - pre >= 0:
                            seg = self.df[sig].values[idx - pre:idx + post]
                            segs.append(seg - np.mean(seg[:pre]))
                    if segs:
                        arr = np.array(segs)
                        mn = np.mean(arr, axis=0); se = stats.sem(arr, axis=0)
                        tv = np.linspace(-2, 8, len(mn))
                        ax.plot(tv, mn, color=color, label=f'{stim_type} (n={len(segs)})')
                        ax.fill_between(tv, mn - se, mn + se, color=color, alpha=0.15)
                        ax.axvline(0, color='k', ls='--', alpha=0.5); ax.axhline(0, color='k', lw=0.5)
                        ax.set_title(f'{phase} \u2014 {ylab}')
                        if i == 1: ax.set_xlabel('Time from Stimulus (s)')
                        ax.set_ylabel('Change from Baseline')
                        ax.legend(loc='upper right', frameon=True, fontsize=9)
        _caption(fig, "Figure 9. Average Event-Related Responses \u00b1 SEM (phasic EDA via biophysical deconvolution â€” sinusoidal filter artefact removed). Blue: CS+, Green: CS\u2212, Red: US.")
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig9_AverageResponses.png', dpi=300); plt.close()

    def plot_fig10_rppg_diagnostics(self):
        if not self.rppg_available or not hasattr(self, '_rppg_raw'): return
        methods = [m for m in ['Green', 'CHROM', 'POS'] if m in self._rppg_raw]
        if not methods: return
        fig = plt.figure(figsize=(18, 22))
        gs = fig.add_gridspec(5, 3, hspace=0.45, wspace=0.35)
        ft = self._rppg_ft; fs_v = self._rppg_fs; t_v = ft / 60

        for j, m in enumerate(methods[:3]):
            ax = fig.add_subplot(gs[0, j])
            sig = self._rppg_raw[m]
            b, a = butter(2, [0.7 / (fs_v / 2), 2.5 / (fs_v / 2)], 'band')
            sig_f = filtfilt(b, a, sig - sig.mean())
            ax.plot(t_v[:len(sig_f)], sig_f, color=METHOD_COL[m], lw=0.8)
            ax.set_title(f'{m} Raw Signal (Bandpassed)')
            ax.set_xlabel('Time (min)'); ax.set_ylabel('Amplitude (AU)')

        ax_psd = fig.add_subplot(gs[1, :2])
        for m in methods:
            sig = self._rppg_raw[m]
            b, a = butter(2, [0.5 / (fs_v / 2), 3.0 / (fs_v / 2)], 'band')
            sig_f = filtfilt(b, a, sig - sig.mean())
            f_psd, p_psd = welch(sig_f, fs=fs_v, nperseg=min(len(sig_f) // 2, int(10 * fs_v)))
            ax_psd.semilogy(f_psd * 60, p_psd, color=METHOD_COL[m], lw=1.5, label=m)
        ax_psd.axvspan(40, 180, color='green', alpha=0.08, label='Normal HR range (40\u2013180 BPM)')
        ax_psd.set_title('PSD Comparison \u2014 All rPPG Methods')
        ax_psd.set_xlabel('Frequency (BPM)'); ax_psd.set_ylabel('Power (AU\u00b2/Hz)')
        ax_psd.set_xlim(20, 250); ax_psd.legend(fontsize=9, frameon=True)

        ax_snr = fig.add_subplot(gs[1, 2])
        snr_vals = []
        hr_lo, hr_hi = (max(0.5, (np.nanmean(self.df['HR'].values) - 15) / 60),
                        min(fs_v / 2 - 0.1, (np.nanmean(self.df['HR'].values) + 15) / 60)) \
            if 'HR' in self.df.columns else (0.8, 2.5)
        for m in methods:
            sig = self._rppg_raw[m]
            b, a = butter(2, [0.5 / (fs_v / 2), 3.0 / (fs_v / 2)], 'band')
            sig_f = filtfilt(b, a, sig - sig.mean())
            f_p, p_p = welch(sig_f, fs=fs_v, nperseg=min(len(sig_f) // 2, int(10 * fs_v)))
            sig_power = trapezoid(p_p[(f_p >= hr_lo) & (f_p <= hr_hi)], f_p[(f_p >= hr_lo) & (f_p <= hr_hi)])
            snr_db = 10 * np.log10(sig_power / (trapezoid(p_p, f_p) + 1e-12) + 1e-12)
            snr_vals.append(snr_db)
        bars = ax_snr.bar(methods[:len(snr_vals)], snr_vals,
                          color=[METHOD_COL[m] for m in methods[:len(snr_vals)]])
        ax_snr.set_title('Signal SNR in HR Band'); ax_snr.set_ylabel('SNR (dB)'); ax_snr.set_xlabel('Method')
        for bar, val in zip(bars, snr_vals):
            ax_snr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9)

        if 'HR' in self.df.columns:
            all_methods = [m for m in ['Green', 'CHROM', 'POS', 'Adaptive'] if f'HR_{m}' in self.df.columns]
            ecg_hr = self.df['HR'].values; t_ph = self.df['time_sec'].values / 60
            ax_err = fig.add_subplot(gs[2, :])
            win_err = int(30 * self.fs)
            for m in all_methods:
                mhr = self.df[f'HR_{m}'].values; err = []; t_err = []
                for s in range(0, len(ecg_hr) - win_err, win_err // 2):
                    seg_e = ecg_hr[s:s + win_err]; seg_m = mhr[s:s + win_err]
                    v = np.isfinite(seg_e) & np.isfinite(seg_m)
                    if v.sum() > 10:
                        err.append(np.sqrt(np.mean((seg_e[v] - seg_m[v]) ** 2)))
                        t_err.append(t_ph[s + win_err // 2])
                if err:
                    ax_err.plot(t_err, err, '-o', color=METHOD_COL.get(m, 'gray'), lw=1.5, ms=4, label=m)
            ax_err.set_title('Temporal RMSE vs ECG (30s rolling windows)')
            ax_err.set_xlabel('Time (min)'); ax_err.set_ylabel('RMSE (BPM)')
            ax_err.legend(fontsize=9, frameon=True); self._phase_shading(ax_err)

            ba_methods = [m for m in ['Green', 'CHROM', 'POS', 'Adaptive'] if f'HR_{m}' in self.df.columns][:3]
            for j, m in enumerate(ba_methods):
                ax = fig.add_subplot(gs[3, j])
                mhr = self.df[f'HR_{m}'].values
                valid = np.isfinite(ecg_hr) & np.isfinite(mhr)
                if valid.sum() > 20:
                    _bland_altman(ax, ecg_hr[valid], mhr[valid], m, METHOD_COL.get(m, 'gray'))

            ax_rank = fig.add_subplot(gs[4, :2])
            rank_data = []
            for m in ['Green', 'CHROM', 'POS', 'Adaptive']:
                if f'HR_{m}' not in self.df.columns: continue
                mhr = self.df[f'HR_{m}'].values
                valid = np.isfinite(ecg_hr) & np.isfinite(mhr)
                if valid.sum() < 20: continue
                r, _ = stats.pearsonr(ecg_hr[valid], mhr[valid])
                rmse = np.sqrt(np.mean((ecg_hr[valid] - mhr[valid]) ** 2))
                bias = np.mean(ecg_hr[valid] - mhr[valid])
                rank_data.append({'Method': m, 'r': f'{r:.3f}', 'RMSE': f'{rmse:.1f}', 'Bias': f'{bias:.1f}'})
            if rank_data:
                df_rank = pd.DataFrame(rank_data)
                ax_rank.axis('off')
                tbl = ax_rank.table(cellText=df_rank.values, colLabels=df_rank.columns,
                                    cellLoc='center', loc='center', colColours=['#f0f0f0'] * 4)
                tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
                ax_rank.set_title('Method Performance Summary (vs ECG Ground Truth)', pad=15)

            ax_adapt = fig.add_subplot(gs[4, 2])
            if 'HR_Adaptive' in self.df.columns and 'HR_CHROM' in self.df.columns:
                chrom = self.df['HR_CHROM'].values; adapted = self.df['HR_Adaptive'].values
                v1 = np.isfinite(ecg_hr) & np.isfinite(chrom)
                v2 = np.isfinite(ecg_hr) & np.isfinite(adapted)
                if v1.sum() > 20 and v2.sum() > 20:
                    r1, _ = stats.pearsonr(ecg_hr[v1], chrom[v1])
                    r2, _ = stats.pearsonr(ecg_hr[v2], adapted[v2])
                    ax_adapt.bar(['CHROM\n(pre-adapt)', 'Adaptive\n(post-adapt)'],
                                 [r1, r2], color=['#FF9800', '#F44336'])
                    ax_adapt.set_title('Adaptive Filter Improvement')
                    ax_adapt.set_ylabel('Pearson r (vs ECG)'); ax_adapt.set_ylim(0, 1)
                    for i, v in enumerate([r1, r2]):
                        ax_adapt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax_adapt.axis('off')
            else:
                ax_adapt.axis('off')

        _caption(fig, "Figure 10. rPPG Diagnostic Suite. (Row 1) Bandpassed raw signal per method. (Row 2) PSD comparison and SNR in HR band. (Row 3) Temporal RMSE vs ECG in 30s windows. (Row 4) Bland-Altman plots. (Row 5) Method ranking table and adaptive filter improvement.")
        plt.savefig(f'{self.output_prefix}_Fig10_rPPG_Diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_fig11_ecg_qrs(self):
        """Figure 11. ECG QRS Diagnostic.
        Three panels: (A) Full-session raw ECG with R-peak markers,
        (B) 10-second zoom centred on median R-peak showing QRS morphology,
        (C) Superimposed beat templates (mean ± SD) for QRS quality assessment.
        """
        if 'ECG_clean' not in self.df.columns:
            return

        ecg = self.df['ECG_clean'].values
        t_s = self.df['time_sec'].values
        fs = self.fs
        rp = (
            np.where(self.df.get('ECG_peaks', pd.Series(0, index=self.df.index)).values == 1)[0]
            if 'ECG_peaks' in self.df.columns else np.array([], dtype=int)
        )

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # ── Panel A: full-session ECG ──────────────────────────────────────
        t_min = t_s / 60
        axes[0].plot(t_min, ecg, 'k-', lw=0.3, alpha=0.8)
        if len(rp):
            axes[0].plot(
                t_min[rp], ecg[rp], 'r|', ms=8, markeredgewidth=1.2,
                label=f'R-peaks (n={len(rp)})'
            )
        axes[0].set_title('A) Full-Session ECG with R-Peak Detection')
        axes[0].set_xlabel('Time (min)')
        axes[0].set_ylabel('Amplitude (AU)')
        axes[0].legend(fontsize=9, frameon=True)
        self._phase_shading(axes[0])

        # ── Panel B: 10-second QRS zoom ────────────────────────────────────
        if len(rp) > 5:
            centre_idx = int(rp[len(rp) // 2])
        else:
            centre_idx = len(ecg) // 2

        half_w = int(5 * fs)
        i0 = max(0, centre_idx - half_w)
        i1 = min(len(ecg), centre_idx + half_w)
        t_zoom = t_s[i0:i1]
        ecg_zoom = ecg[i0:i1]

        axes[1].plot(t_zoom, ecg_zoom, 'k-', lw=1.2)
        rp_zoom = rp[(rp >= i0) & (rp < i1)]
        if len(rp_zoom):
            axes[1].plot(t_s[rp_zoom], ecg[rp_zoom], 'rv', ms=7, label='R-peaks')
        axes[1].set_title('B) 10-Second ECG Zoom — QRS Morphology')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (AU)')
        axes[1].legend(fontsize=9, frameon=True)

        # ── Panel C: superimposed beat templates ───────────────────────────
        pre_s = int(0.15 * fs)   # 150 ms pre-R
        post_s = int(0.35 * fs)  # 350 ms post-R (full P-QRS-T)
        beats = []

        for r in rp:
            if r - pre_s >= 0 and r + post_s < len(ecg):
                seg = ecg[r - pre_s:r + post_s]
                seg = seg - seg.mean()
                beats.append(seg)

        if len(beats) >= 5:
            beats_arr = np.array(beats)
            t_beat_ms = np.linspace(-pre_s / fs * 1000, post_s / fs * 1000, beats_arr.shape[1])
            beat_mean = beats_arr.mean(axis=0)
            beat_sd = beats_arr.std(axis=0)

            # Plot individual beats faintly
            for b in beats_arr[::max(1, len(beats_arr)//40)]:
                axes[2].plot(t_beat_ms, b, 'steelblue', lw=0.4, alpha=0.25)

            axes[2].plot(t_beat_ms, beat_mean, 'k-', lw=2.5, label='Mean')
            axes[2].fill_between(
                t_beat_ms,
                beat_mean - beat_sd,
                beat_mean + beat_sd,
                color='steelblue', alpha=0.25, label='±1 SD'
            )
            axes[2].axvline(0, color='r', ls='--', lw=1, label='R-peak (t=0)')
            axes[2].set_title(f'C) Beat Template Overlay (n={len(beats)} beats) — QRS Quality')
            axes[2].set_xlabel('Time from R-peak (ms)')
            axes[2].set_ylabel('Amplitude (AU, mean-subtracted)')
            axes[2].legend(fontsize=9, frameon=True)

            # SNR annotation
            qrs_mask = (t_beat_ms >= -50) & (t_beat_ms <= 100)
            noise_mask = ~qrs_mask
            if qrs_mask.sum() > 0 and noise_mask.sum() > 0:
                snr = (
                    (beat_mean[qrs_mask].max() - beat_mean[qrs_mask].min()) /
                    (beat_sd[noise_mask].mean() + 1e-9)
                )
                axes[2].text(
                    0.98, 0.95, f'QRS SNR ≈ {snr:.1f}',
                    transform=axes[2].transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
                )
        else:
            axes[2].text(
                0.5, 0.5, 'Insufficient beats for template',
                ha='center', va='center', transform=axes[2].transAxes
            )
            axes[2].axis('off')

        _caption(
            fig,
            "Figure 11. ECG QRS Diagnostic. "
            "(A) Full session ECG with R-peak markers. "
            "(B) 10-second zoom centred on session midpoint showing individual QRS complexes. "
            "(C) Superimposed beat templates: individual beats (blue, semi-transparent), "
            "mean waveform (black), ±1 SD (shaded). QRS SNR = peak QRS amplitude / baseline noise SD."
        )
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        plt.savefig(f'{self.output_prefix}_Fig11_ECG_QRS.png', dpi=300)
        plt.close()

        # -----------------------------------------------------------------------
    def run(self):
        def _try(fn, label):
            try:
                fn()
            except Exception as e:
                import traceback
                print(f"  \u26a0 Skipping {label}: {e}"); traceback.print_exc()

        _try(self.process_signals, "process_signals")
        _try(self.detect_phases, "detect_phases")
        _try(self.run_glm_analysis, "run_glm_analysis")
        _try(self.compute_metrics, "compute_metrics")
        _try(self.process_rppg, "process_rppg")

        print("Generating figures v7.16...")
        for fn, label in [
            (self.plot_fig1_overview, "Fig1_Overview"),
            (self.plot_fig2_hrv, "Fig2_HRV"),
            (self.plot_fig3_eda_glm, "Fig3_EDA_GLM"),
            (self.plot_fig4_spectral, "Fig4_Spectral"),
            (self.plot_fig5_snr, "Fig5_SNR"),
            (self.plot_fig6_heatmaps, "Fig6_Heatmaps"),
            (self.plot_fig7_rppg_comparison, "Fig7_rPPG_Comparison"),
            (self.plot_fig8_zooms, "Fig8_Zooms"),
            (self.plot_fig9_averages, "Fig9_AverageResponses"),
            (self.plot_fig10_rppg_diagnostics, "Fig10_rPPG_Diagnostics"),
            (self.plot_fig11_ecg_qrs,           "Fig11_ECG_QRS"),
        ]:
            _try(fn, label)

        rows = []
        if self.glm_results_eda:
            for k, v in self.glm_results_eda['betas'].items():
                rows.append({'metric': f'Beta_{k}', 'value': v})
        pd.DataFrame(rows).to_csv(f'{self.output_prefix}_metrics_summary.csv', index=False)
        print("Done.")
        print("=" * 70)


# ============================================================================
# CLI
# ============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--physio', required=True)
    p.add_argument('--markers', required=True)
    p.add_argument('--video', default=None)
    p.add_argument('--frame-timing', default=None)
    p.add_argument('--output-dir', default='analysis_output')
    args = p.parse_args()
    cfg = Config()
    cfg.PHYSIO_CSV = args.physio; cfg.MARKERS_CSV = args.markers
    cfg.VIDEO_PATH = args.video; cfg.FRAME_TIMING_CSV = args.frame_timing
    cfg.OUTPUT_DIR = args.output_dir
    UnifiedPhysioPipeline(config=cfg).run()

