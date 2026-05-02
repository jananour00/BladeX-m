"""
Gait Classifier Preprocessor
==============================
Handles DSP pipeline for the 3-class gait model:
  - Healthy / Amputee K2 / Amputee K3

The model was trained on:
  • 9 biomechanical signals (angles, GRF, moments)
  • Each normalized to 101 points
  • 23 statistical features extracted per signal → 207 features
  • + 3 spatiotemporal features (step_width, cycle_duration, velocity)
  • Total = 210 features

HOW TO CONNECT NEW INPUT TO THIS PREPROCESSOR:
  1. Collect raw signals (list of lists) from CSV upload or manual input
  2. Call: features = GaitPreprocessor.extract_features_from_signals(signals, spatiotemporal)
  3. Pass features to model.predict(features)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions
_trapz = getattr(np, 'trapezoid', None) or np.trapz


class GaitPreprocessor:
    """
    All DSP + feature extraction logic for the Gait Classifier.
    Stateless — all methods are static so no instantiation needed.
    """

    N_POINTS = 101          # All signals normalized to this length
    N_FEATURES_PER_SIGNAL = 23   # Features extracted per signal
    N_SIGNALS = 9           # Expected signals per subject
    N_SPATIOTEMPORAL = 3    # step_width, cycle_duration, velocity
    TOTAL_FEATURES = N_SIGNALS * N_FEATURES_PER_SIGNAL + N_SPATIOTEMPORAL  # 210

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Normalize any signal to exactly 101 points
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def normalize_to_101_points(signal: list) -> np.ndarray:
        """
        Interpolates any-length signal to exactly 101 gait-cycle points.
        Handles NaN values by linear interpolation before cubic resampling.

        Returns: np.ndarray of shape (101,)
        """
        s = np.array(signal, dtype=float)
        valid = ~np.isnan(s)

        if valid.sum() < 5:
            # Not enough valid data points
            return np.full(GaitPreprocessor.N_POINTS, np.nan)

        # Fill NaN gaps by linear interpolation
        if valid.sum() < len(s):
            x = np.arange(len(s))
            s = np.interp(x, x[valid], s[valid])

        # Cubic resample to 101 points
        original_idx = np.linspace(0, 1, len(s))
        target_idx = np.linspace(0, 1, GaitPreprocessor.N_POINTS)
        f = interp1d(original_idx, s, kind='cubic', fill_value='extrapolate')
        return f(target_idx)

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Extract 23 statistical features from one normalized signal
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def extract_signal_features(signal_normalized: np.ndarray) -> list:
        """
        Extracts 23 statistical + biomechanical features from a 101-point signal.

        Feature breakdown:
          1-6:   Basic statistics (mean, std, max, min, ROM, area)
          7-10:  Peak/trough positions and values
          11-14: Slope features (mean, max, min, variance)
          15-16: Energy and spectral entropy
          17-18: Symmetry and zero-crossing rate
          19-23: Percentile values (10th, 25th, 50th, 75th, 90th)

        Returns: list of 23 floats (NaN if signal is invalid)
        """
        s = signal_normalized

        if np.any(np.isnan(s)):
            return [np.nan] * GaitPreprocessor.N_FEATURES_PER_SIGNAL

        # ── Basic statistics ──────────────────────────────────────
        mean_val = float(np.mean(s))
        std_val  = float(np.std(s))
        max_val  = float(np.max(s))
        min_val  = float(np.min(s))
        rom      = max_val - min_val        # Range of Motion
        area     = float(_trapz(s))          # Area under curve

        # ── Peaks and troughs (percentile-based, no hardcoded indices) ──
        peaks,   _ = find_peaks(s,  height=np.mean(s), distance=10)
        troughs, _ = find_peaks(-s, distance=10)

        first_peak_idx   = float(peaks[0]   / len(s)) if len(peaks)   > 0 else 0.5
        peak_value       = float(s[peaks[0]])          if len(peaks)   > 0 else max_val
        first_trough_idx = float(troughs[0] / len(s)) if len(troughs) > 0 else 0.25
        trough_value     = float(s[troughs[0]])         if len(troughs) > 0 else min_val

        # ── Slope (rate of change) ────────────────────────────────
        slopes        = np.diff(s)
        mean_slope    = float(np.mean(slopes))
        max_slope     = float(np.max(slopes))
        min_slope     = float(np.min(slopes))
        slope_variance= float(np.var(slopes))

        # ── Energy and entropy ────────────────────────────────────
        energy = float(np.sum(s ** 2) / len(s))
        hist, _ = np.histogram(s, bins=10)
        hist_norm = hist[hist > 0] / len(s)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm))) if len(hist_norm) > 0 else 0.0

        # ── Gait symmetry (first half vs second half correlation) ──
        half = len(s) // 2
        if half > 1:
            symmetry = float(np.corrcoef(s[:half], s[half:half * 2])[0, 1])
            if np.isnan(symmetry):
                symmetry = 0.0
        else:
            symmetry = 0.0

        # ── Zero-crossing rate ────────────────────────────────────
        zc = np.sum(np.diff(np.signbit(s - np.mean(s))))
        zcr = float(zc / len(s))

        # ── Percentile values ─────────────────────────────────────
        pct_values = [float(np.percentile(s, p)) for p in [10, 25, 50, 75, 90]]

        # Assemble: 18 base + 5 percentiles = 23 features
        features = [
            mean_val, std_val, max_val, min_val, rom, area,
            first_peak_idx, peak_value, first_trough_idx, trough_value,
            mean_slope, max_slope, min_slope, slope_variance,
            energy, entropy, symmetry, zcr,
        ] + pct_values

        return features  # length == 23

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Process all 9 signals for one subject
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def extract_features_from_signals(
        signals: list,
        spatiotemporal: dict = None
    ) -> np.ndarray:
        """
        Main entry point: converts raw signals → model-ready feature vector.

        Args:
            signals: list of 9 raw signal arrays (each can be any length)
            spatiotemporal: dict with optional keys:
                'step_width', 'cycle_duration', 'velocity'

        Returns:
            np.ndarray of shape (1, 210) — ready for model.predict()

        HOW TO USE IN FLASK ROUTE:
            features = GaitPreprocessor.extract_features_from_signals(
                signals=[[...], [...], ...],   # 9 lists from CSV
                spatiotemporal={'step_width': 0.12, 'cycle_duration': 1.1, 'velocity': 1.3}
            )
            prediction = gait_model.predict(features)
        """
        all_features = []

        # Pad to 9 signals if fewer provided
        while len(signals) < GaitPreprocessor.N_SIGNALS:
            signals.append(np.full(50, np.nan).tolist())

        # Process each signal
        for signal in signals[:GaitPreprocessor.N_SIGNALS]:
            normalized = GaitPreprocessor.normalize_to_101_points(signal)
            features   = GaitPreprocessor.extract_signal_features(normalized)
            all_features.extend(features)

        # Append spatiotemporal features
        if spatiotemporal is None:
            spatiotemporal = {}

        all_features.append(float(spatiotemporal.get('step_width',     np.nan)))
        all_features.append(float(spatiotemporal.get('cycle_duration', np.nan)))
        all_features.append(float(spatiotemporal.get('velocity',       np.nan)))

        return np.array(all_features, dtype=float).reshape(1, -1)

    # ──────────────────────────────────────────────────────────────────
    # UTILITY: Parse CSV upload into signals
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def parse_csv_to_signals(df) -> dict:
        """
        Parses a pandas DataFrame from CSV upload into signals dict.

        Expected CSV columns (flexible — maps by name):
          ankle_angle, knee_angle, hip_angle,
          grf_vertical, grf_ap, grf_ml,
          ankle_moment, knee_moment, hip_moment,
          step_width (optional), cycle_duration (optional), velocity (optional)

        Returns:
            {
              'signals': [[...], [...], ...],   # 9 lists
              'spatiotemporal': {...},
              'signal_labels': [...]            # for chart display
            }

        HOW TO EXTEND:
          Add more column mappings here to accept different CSV formats.
        """
        SIGNAL_MAP = {
            'ankle_angle':  ['ankle_angle', 'ankle_fe', 'afe', 'n_afe'],
            'knee_angle':   ['knee_angle',  'knee_fe',  'kfe', 'n_kfe'],
            'hip_angle':    ['hip_angle',   'hip_fe',   'hipfe', 'n_hipfe'],
            'grf_vertical': ['grf_vertical', 'grf_v', 'vertical_grf', 'n_grf_vertical'],
            'grf_ap':       ['grf_ap', 'ap_grf', 'n_grf_ap'],
            'grf_ml':       ['grf_ml', 'ml_grf', 'n_grf_ml'],
            'ankle_moment': ['ankle_moment', 'moment_afe', 'n_moment_afe'],
            'knee_moment':  ['knee_moment',  'moment_kfe', 'n_moment_kfe'],
            'hip_moment':   ['hip_moment',   'moment_hipfe', 'n_moment_hipfe'],
        }

        SPATIO_MAP = {
            'step_width':     ['step_width', 'stepwidth'],
            'cycle_duration': ['cycle_duration', 'gait_cycle', 'cycle'],
            'velocity':       ['velocity', 'speed', 'walking_speed'],
        }

        # Normalize column names to lowercase
        cols_lower = {c.lower().replace(' ', '_'): c for c in df.columns}

        signals = []
        signal_labels = []

        for signal_name, aliases in SIGNAL_MAP.items():
            found = False
            for alias in aliases:
                if alias in cols_lower:
                    col = cols_lower[alias]
                    signals.append(df[col].dropna().tolist())
                    signal_labels.append(signal_name.replace('_', ' ').title())
                    found = True
                    break
            if not found:
                signals.append([0.0] * 50)  # placeholder
                signal_labels.append(signal_name.replace('_', ' ').title() + ' (missing)')

        spatiotemporal = {}
        for key, aliases in SPATIO_MAP.items():
            for alias in aliases:
                if alias in cols_lower:
                    col = cols_lower[alias]
                    val = df[col].dropna().mean()
                    spatiotemporal[key] = float(val) if not np.isnan(val) else np.nan
                    break

        return {
            'signals': signals,
            'spatiotemporal': spatiotemporal,
            'signal_labels': signal_labels
        }