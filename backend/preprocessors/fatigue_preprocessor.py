"""
Fatigue & Injury Preprocessor
================================
Handles feature engineering for the 2-pipeline model:
  - fatigue_pipeline.joblib   → GradientBoostingRegressor  → predicts fatigue score (0–1)
  - injury_pipeline.joblib    → GradientBoostingClassifier → binary injury risk (0/1)
 
The model was trained on REAL_PARALYMPIC_DATASET_40_RUNNERS.csv
(Paralympic runners, NOT the same as gait classifier data).
 
FEATURES NEEDED:
  Fatigue model  (21 features): see FEATURES_FATIGUE below
  Injury model   (20 features): see FEATURES_INJURY below
 
RISK THRESHOLDS (from thresholds.json):
  fatigue_threshold : 0.897   → above this = HIGH fatigue
  asym_threshold    : 0.979   → above this = HIGH asymmetry → injury risk
 
HOW TO CONNECT NEW INPUT:
  1. User provides CSV or manual values matching the feature columns
  2. Call: FatiguePreprocessor.prepare_input(data_dict)
  3. Pass result to:
       fatigue_score   = fatigue_pipeline.predict(X_fatigue)
       injury_risk     = injury_pipeline.predict_proba(X_injury)
  4. Compare against thresholds to determine risk level
"""
 
import numpy as np
import pandas as pd
 
 
# ──────────────────────────────────────────────────────────────────────────
# Feature lists (must match training order exactly)
# ──────────────────────────────────────────────────────────────────────────
FEATURES_FATIGUE = [
    "speed", "stride_length", "cadence",
    "knee_prosthetic", "knee_sound",
    "hip_prosthetic", "hip_sound",
    "asymmetry_ratio", "asymmetry_stride", "symmetry_decay",
    "bmi", "peak_speed_ms", "variability",
    "roll_speed_mean", "roll_speed_std",
    "roll_knee_std", "roll_hip_std",
    "roll_variability_std",
    "roll_knee_std_w5", "roll_hip_std_w5",
    "cumulative_impact",
]
 
FEATURES_INJURY = [
    "speed", "stride_length", "cadence",
    "knee_prosthetic", "knee_sound",
    "hip_prosthetic", "hip_sound",
    "asymmetry_stride", "symmetry_decay",
    "bmi", "peak_speed_ms", "variability",
    "roll_speed_mean", "roll_speed_std",
    "roll_knee_std", "roll_hip_std",
    "roll_knee_std_w5", "roll_hip_std_w5",
    "roll_variability_std",
    "cumulative_impact",
]
 
# Thresholds (loaded from thresholds.json at startup)
DEFAULT_THRESHOLDS = {
    "fatigue_threshold": 0.897,
    "asym_threshold": 0.9788,
}
 
 
class FatiguePreprocessor:
    """
    Feature engineering for Fatigue & Injury models.
    All methods are static — no state, easy to import anywhere.
    """
 
    # ──────────────────────────────────────────────────────────────────
    # MAIN ENTRY: From a dict of raw values → model-ready arrays
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def prepare_input(data: dict) -> dict:
        """
        Prepares model inputs from raw user-provided data.
 
        Args:
            data: dict with any subset of the raw columns:
              Required (ideally): speed, stride_length, cadence,
                knee_left, knee_right, hip_left, hip_right,
                prosthetic_side ('left' or 'right'),
                weight_kg, height_cm, peak_speed_ms, variability,
                asymmetry_stride
 
              Optional (computed if missing):
                roll_* (set to 0 if only one row provided)
                cumulative_impact, symmetry_decay, asymmetry_ratio, bmi
 
        Returns:
            {
              'X_fatigue': np.ndarray shape (1, 21),
              'X_injury':  np.ndarray shape (1, 20),
              'computed':  dict of all derived features (for display)
            }
 
        HOW TO EXTEND:
          Add more raw-to-feature mappings in _engineer_features() below.
        """
        engineered = FatiguePreprocessor._engineer_features(data)
        computed = dict(engineered)
 
        X_fatigue = np.array(
            [float(engineered.get(f, np.nan)) for f in FEATURES_FATIGUE],
            dtype=float
        ).reshape(1, -1)
 
        X_injury = np.array(
            [float(engineered.get(f, np.nan)) for f in FEATURES_INJURY],
            dtype=float
        ).reshape(1, -1)
 
        return {
            'X_fatigue': X_fatigue,
            'X_injury':  X_injury,
            'computed':  computed
        }
 
    # ──────────────────────────────────────────────────────────────────
    # Feature Engineering (mirrors the notebook's load_and_engineer)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _engineer_features(data: dict) -> dict:
        """
        Derives all model features from raw input.
 
        NOTE: Rolling features (roll_*) require time-series data.
        For single-row manual input, they default to 0 (no history).
        For CSV upload with multiple rows, they are computed properly.
        """
        out = dict(data)  # Start with raw values
 
        # ── Prosthetic side decomposition ────────────────────────────
        side = str(data.get('prosthetic_side', 'right')).lower()
 
        knee_left  = float(data.get('knee_left',  data.get('knee_prosthetic', np.nan)))
        knee_right = float(data.get('knee_right', data.get('knee_sound',      np.nan)))
        hip_left   = float(data.get('hip_left',   data.get('hip_prosthetic',  np.nan)))
        hip_right  = float(data.get('hip_right',  data.get('hip_sound',       np.nan)))
 
        if side == 'right':
            out['knee_prosthetic'] = knee_right
            out['knee_sound']      = knee_left
            out['hip_prosthetic']  = hip_right
            out['hip_sound']       = hip_left
        else:
            out['knee_prosthetic'] = knee_left
            out['knee_sound']      = knee_right
            out['hip_prosthetic']  = hip_left
            out['hip_sound']       = hip_right
 
        # ── Derived features ─────────────────────────────────────────
        kp = out.get('knee_prosthetic', np.nan)
        ks = out.get('knee_sound',      np.nan)
 
        out['asymmetry_ratio'] = float(ks) / (float(kp) + 1e-6) if not np.isnan(kp) else np.nan
 
        weight = float(data.get('weight_kg', np.nan))
        height = float(data.get('height_cm', np.nan))
        if not (np.isnan(weight) or np.isnan(height)) and height > 0:
            out['bmi'] = weight / ((height / 100) ** 2)
        else:
            out['bmi'] = np.nan
 
        # ── Rolling features: 0 for single-row input ─────────────────
        # These are meaningful only when processing a time-series CSV.
        # For manual input, 0 means "no history available".
        for col in ['roll_speed_mean', 'roll_speed_std',
                    'roll_knee_std', 'roll_hip_std',
                    'roll_knee_std_w5', 'roll_hip_std_w5',
                    'roll_variability_std']:
            if col not in out:
                out[col] = 0.0
 
        if 'cumulative_impact' not in out:
            out['cumulative_impact'] = 0.0
 
        if 'symmetry_decay' not in out:
            out['symmetry_decay'] = 0.0
 
        if 'asymmetry_stride' not in out:
            kp_v = out.get('knee_prosthetic', np.nan)
            ks_v = out.get('knee_sound', np.nan)
            out['asymmetry_stride'] = abs(float(kp_v) - float(ks_v)) if not (np.isnan(kp_v) or np.isnan(ks_v)) else 0.0
 
        return out
 
    # ──────────────────────────────────────────────────────────────────
    # Parse CSV upload → list of input dicts (one per row)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def parse_csv(df: pd.DataFrame) -> list:
        """
        Converts a DataFrame from CSV upload to list of feature dicts
        with rolling features properly computed over time.
 
        Args:
            df: DataFrame with raw columns
 
        Returns:
            list of dicts, each ready for prepare_input()
 
        HOW TO EXTEND:
          Add more column aliases in col_map below.
        """
        # Normalize column names
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
 
        col_map = {
            'speed':          ['speed', 'walking_speed', 'velocity'],
            'stride_length':  ['stride_length', 'step_length'],
            'cadence':        ['cadence', 'steps_per_minute'],
            'knee_left':      ['knee_left', 'knee_l'],
            'knee_right':     ['knee_right', 'knee_r'],
            'hip_left':       ['hip_left', 'hip_l'],
            'hip_right':      ['hip_right', 'hip_r'],
            'weight_kg':      ['weight_kg', 'weight', 'mass'],
            'height_cm':      ['height_cm', 'height'],
            'peak_speed_ms':  ['peak_speed_ms', 'peak_speed', 'max_speed'],
            'variability':    ['variability', 'step_variability'],
            'asymmetry_stride': ['asymmetry_stride'],
            'prosthetic_side':  ['prosthetic_side', 'side'],
        }
 
        # Apply mapping
        for target, aliases in col_map.items():
            if target not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[target] = df[alias]
                        break
 
        # Fill missing columns
        for col in ['prosthetic_side']:
            if col not in df.columns:
                df[col] = 'right'
 
        # Compute rolling features
        W, W5, W50 = 20, 5, 50
 
        if 'speed' in df.columns:
            df['roll_speed_mean'] = df['speed'].rolling(W, min_periods=1).mean()
            df['roll_speed_std']  = df['speed'].rolling(W, min_periods=1).std().fillna(0)
 
        # Prosthetic vs sound decomposition (assume one prosthetic_side per file)
        side = str(df['prosthetic_side'].iloc[0]).lower() if 'prosthetic_side' in df.columns else 'right'
 
        if 'knee_right' in df.columns and 'knee_left' in df.columns:
            if side == 'right':
                df['knee_prosthetic'] = df['knee_right']
                df['knee_sound']      = df['knee_left']
                df['hip_prosthetic']  = df.get('hip_right', pd.Series(np.nan, index=df.index))
                df['hip_sound']       = df.get('hip_left',  pd.Series(np.nan, index=df.index))
            else:
                df['knee_prosthetic'] = df['knee_left']
                df['knee_sound']      = df['knee_right']
                df['hip_prosthetic']  = df.get('hip_left',  pd.Series(np.nan, index=df.index))
                df['hip_sound']       = df.get('hip_right', pd.Series(np.nan, index=df.index))
 
        if 'knee_prosthetic' in df.columns:
            df['roll_knee_std']    = df['knee_prosthetic'].rolling(W,  min_periods=1).std().fillna(0)
            df['roll_knee_std_w5'] = df['knee_prosthetic'].rolling(W5, min_periods=1).std().fillna(0)
        if 'hip_prosthetic' in df.columns:
            df['roll_hip_std']     = df['hip_prosthetic'].rolling(W,  min_periods=1).std().fillna(0)
            df['roll_hip_std_w5']  = df['hip_prosthetic'].rolling(W5, min_periods=1).std().fillna(0)
        if 'variability' in df.columns:
            df['roll_variability_std'] = df['variability'].rolling(W, min_periods=1).std().fillna(0)
 
        # Cumulative impact
        if 'speed' in df.columns and 'weight_kg' in df.columns:
            weight = df['weight_kg'].iloc[0]
            dt = df.index.to_series().diff().fillna(0)
            ci = (df['speed'] * weight * dt).cumsum()
            max_ci = ci.max()
            df['cumulative_impact'] = ci / (max_ci + 1e-9) if max_ci > 0 else 0.0
 
        # Symmetry decay
        if 'asymmetry_ratio' not in df.columns and 'knee_prosthetic' in df.columns:
            df['asymmetry_ratio'] = df['knee_sound'] / (df['knee_prosthetic'] + 1e-6)
 
        if 'asymmetry_ratio' in df.columns:
            rolling_mean = df['asymmetry_ratio'].rolling(W50, min_periods=5).mean()
            baseline     = df['asymmetry_ratio'].expanding().mean()
            df['symmetry_decay'] = (rolling_mean - baseline).fillna(0)
 
        # Return as list of dicts
        return df.to_dict(orient='records')
 
    # ──────────────────────────────────────────────────────────────────
    # Risk level interpretation
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def interpret_fatigue(fatigue_score: float, threshold: float) -> dict:
        """
        Converts raw fatigue score to risk level + display info.
 
        Returns:
            {
              'level': 'Low' | 'Moderate' | 'High',
              'color': CSS color string,
              'icon':  emoji,
              'message': clinical interpretation,
              'score': 0-100 percentage
            }
        """
        score_pct = round(float(fatigue_score) * 100, 1)
 
        if fatigue_score < threshold * 0.7:
            return {
                'level': 'Low',
                'color': '#22c55e',
                'icon': '✅',
                'message': 'Fatigue levels within normal range. Performance maintained.',
                'score': score_pct,
                'threshold_pct': round(threshold * 100, 1)
            }
        elif fatigue_score < threshold:
            return {
                'level': 'Moderate',
                'color': '#f59e0b',
                'icon': '⚠️',
                'message': 'Approaching fatigue threshold. Monitor closely.',
                'score': score_pct,
                'threshold_pct': round(threshold * 100, 1)
            }
        else:
            return {
                'level': 'High',
                'color': '#ef4444',
                'icon': '🚨',
                'message': 'High fatigue detected. Recommend rest or intervention.',
                'score': score_pct,
                'threshold_pct': round(threshold * 100, 1)
            }
 
    @staticmethod
    def interpret_injury(injury_prob: float, asym_ratio: float, asym_threshold: float) -> dict:
        """
        Converts injury probability + asymmetry to risk level.
 
        Returns:
            {
              'level': 'Low' | 'Moderate' | 'High',
              'probability_pct': float,
              'asymmetry_flag': bool,
              'message': str
            }
        """
        prob_pct = round(float(injury_prob) * 100, 1)
        asym_flag = float(asym_ratio) > float(asym_threshold) if not np.isnan(asym_ratio) else False
 
        if injury_prob < 0.3:
            level = 'Low'
            color = '#22c55e'
            msg = 'Low injury risk. Biomechanics appear stable.'
        elif injury_prob < 0.6:
            level = 'Moderate'
            color = '#f59e0b'
            msg = 'Moderate injury risk. Asymmetry patterns warrant attention.'
        else:
            level = 'High'
            color = '#ef4444'
            msg = 'High injury risk. Significant biomechanical imbalance detected.'
 
        return {
            'level': level,
            'color': color,
            'probability_pct': prob_pct,
            'asymmetry_flag': asym_flag,
            'message': msg
        }
 