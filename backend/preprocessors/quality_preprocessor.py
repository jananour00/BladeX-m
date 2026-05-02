"""
Quality of Movement & Coach Assistant Preprocessor
====================================================
Handles feature engineering for the two Quality+Coach models:
  - quality_model.joblib   → RandomForestRegressor  → quality score (0–100)
  - coach_model.joblib     → RandomForestClassifier → performance level (Elite/Advanced/Developing)
 
Models trained on REAL_PARALYMPIC_DATASET_40_RUNNERS.csv
(75,212 samples from 37 Paralympic blade runners)
 
FEATURES:
  Quality model:  avg_speed, peak_speed, avg_cadence, avg_stride_length,
                  avg_asymmetry_knee, max_asymmetry_knee, avg_asymmetry_stride,
                  avg_variability, smoothness, coordination, efficiency
 
  Coach model:    peak_speed, avg_cadence, avg_stride, asymmetry,
                  variability, smoothness, coordination
 
HOW TO CONNECT:
  1. User provides CSV with columns: runner_id (opt), speed, cadence,
     stride_length, asymmetry_knee (or knee_left/right),
     asymmetry_stride, variability, time (opt)
  2. Call QualityPreprocessor.prepare_from_df(df)
  3. Pass to quality_model.predict() and coach_model.predict()
"""
 
import numpy as np
import pandas as pd
 
# Features the quality model was trained on (must match training order)
FEATURES_QUALITY = [
    "avg_speed", "peak_speed", "avg_cadence", "avg_stride_length",
    "avg_asymmetry_knee", "max_asymmetry_knee", "avg_asymmetry_stride",
    "avg_variability", "smoothness", "coordination", "efficiency",
]
 
# Features the coach model was trained on
FEATURES_COACH = [
    "peak_speed", "avg_cadence", "avg_stride", "asymmetry",
    "variability", "smoothness", "coordination",
]
 
PERFORMANCE_LEVELS = {
    0: "Developing",
    1: "Advanced",
    2: "Elite",
}
 
# Thresholds for coach feedback (from notebook)
CADENCE_TARGET = 4.2          # Hz  — below this = flag
STRIDE_TARGET  = 2.3          # m   — below this = flag
ASYMMETRY_TARGET = 7.0        # deg — above this = flag
ELITE_SPEED    = 10.5         # m/s
ADVANCED_SPEED =  9.5         # m/s
 
 
class QualityPreprocessor:
    """
    Feature engineering for Quality of Movement & Coach Assistant models.
    All methods static — stateless, easy to import anywhere.
    """
 
    # ──────────────────────────────────────────────────────────────────
    # Main entry: DataFrame → model-ready feature vectors
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def prepare_from_df(df: pd.DataFrame) -> dict:
        """
        Processes a DataFrame (CSV upload) for both Quality and Coach models.
 
        Args:
            df: DataFrame with columns from the Paralympic dataset.
                Required: speed, cadence, stride_length, variability
                Recommended: asymmetry_knee (or knee_left + knee_right),
                             asymmetry_stride, runner_id, time
 
        Returns:
            {
              'X_quality':  np.ndarray shape (1, 11)  — quality model input
              'X_coach':    np.ndarray shape (1, 7)   — coach model input
              'features':   dict of all engineered features (for display)
              'time_series': dict with per-row time series for charts
              'runner_ids':  list of runner ids (if column present)
            }
        """
        df = QualityPreprocessor._normalize_columns(df)
        df = QualityPreprocessor._engineer_row_features(df)
        agg = QualityPreprocessor._aggregate_features(df)
 
        X_quality = np.array(
            [float(agg.get(f, 0.0)) for f in FEATURES_QUALITY],
            dtype=float
        ).reshape(1, -1)
 
        X_coach = np.array(
            [float(agg.get(f, 0.0)) for f in FEATURES_COACH],
            dtype=float
        ).reshape(1, -1)
 
        # Time series for charts
        time_axis = df['time'].tolist() if 'time' in df.columns else list(range(len(df)))
        time_series = {
            'time': time_axis,
            'speed':        df['speed'].tolist() if 'speed' in df.columns else [],
            'variability':  df['variability'].tolist() if 'variability' in df.columns else [],
            'smoothness':   df['smoothness'].tolist() if 'smoothness' in df.columns else [],
            'coordination': df['coordination_score'].tolist() if 'coordination_score' in df.columns else [],
            'quality_score': df['quality_score'].tolist() if 'quality_score' in df.columns else [],
        }
 
        runner_ids = df['runner_id'].unique().tolist() if 'runner_id' in df.columns else []
 
        return {
            'X_quality':   X_quality,
            'X_coach':     X_coach,
            'features':    agg,
            'time_series': time_series,
            'runner_ids':  runner_ids,
        }
 
    @staticmethod
    def prepare_from_manual(data: dict) -> dict:
        """
        Processes manually entered values for both models.
 
        Args:
            data: dict with keys:
              speed, cadence, stride_length, variability,
              asymmetry_knee (or knee_left + knee_right),
              asymmetry_stride  (optional)
 
        Returns: same schema as prepare_from_df (without time_series)
        """
        # Build a one-row dataframe and run through the same pipeline
        df = pd.DataFrame([data])
        df = QualityPreprocessor._normalize_columns(df)
        df = QualityPreprocessor._engineer_row_features(df)
        agg = QualityPreprocessor._aggregate_features(df)
 
        X_quality = np.array(
            [float(agg.get(f, 0.0)) for f in FEATURES_QUALITY],
            dtype=float
        ).reshape(1, -1)
 
        X_coach = np.array(
            [float(agg.get(f, 0.0)) for f in FEATURES_COACH],
            dtype=float
        ).reshape(1, -1)
 
        return {
            'X_quality': X_quality,
            'X_coach':   X_coach,
            'features':  agg,
        }
 
    # ──────────────────────────────────────────────────────────────────
    # Column normalisation
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
 
        aliases = {
            'speed':           ['speed', 'walking_speed', 'velocity', 'running_speed'],
            'cadence':         ['cadence', 'steps_per_minute', 'step_freq'],
            'stride_length':   ['stride_length', 'step_length', 'avg_stride'],
            'variability':     ['variability', 'stride_variability', 'step_variability'],
            'asymmetry_knee':  ['asymmetry_knee', 'knee_asymmetry', 'knee_diff'],
            'asymmetry_stride':['asymmetry_stride', 'stride_asymmetry'],
            'knee_left':       ['knee_left', 'knee_l'],
            'knee_right':      ['knee_right', 'knee_r'],
            'runner_id':       ['runner_id', 'athlete_id', 'subject_id', 'id'],
            'time':            ['time', 'timestamp', 't'],
        }
 
        for target, srcs in aliases.items():
            if target not in df.columns:
                for src in srcs:
                    if src in df.columns:
                        df[target] = df[src]
                        break
 
        # Derive asymmetry_knee from left/right if missing
        if 'asymmetry_knee' not in df.columns and 'knee_left' in df.columns and 'knee_right' in df.columns:
            df['asymmetry_knee'] = (df['knee_left'] - df['knee_right']).abs()
 
        return df
 
    # ──────────────────────────────────────────────────────────────────
    # Row-level feature engineering (mirrors the notebook)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
 
        # Fill required columns with safe defaults
        for col in ['speed', 'cadence', 'stride_length', 'variability', 'asymmetry_knee', 'asymmetry_stride']:
            if col not in df.columns:
                df[col] = 0.0
 
        # Efficiency score (speed / cadence × stride_length)
        denom = (df['cadence'] * df['stride_length']).replace(0, np.nan)
        df['efficiency_score'] = (df['speed'] / denom).clip(0, 1).fillna(0)
 
        # Jerkiness & smoothness
        speed_vals = df['speed'].fillna(0).values
        if len(speed_vals) > 2:
            accel = np.gradient(speed_vals)
            jerk  = np.abs(np.gradient(accel))
        else:
            jerk = np.zeros(len(speed_vals))
        df['jerkiness'] = jerk
        df['smoothness'] = 1 / (1 + df['jerkiness'] * 10)
 
        # Coordination score
        df['coordination_score'] = (1 - df['asymmetry_knee'] / 20).clip(0, 1)
 
        # Quality score (0–100), matching notebook formula
        df['quality_score'] = (
            df['smoothness'] * 30 +
            df['coordination_score'] * 30 +
            df['efficiency_score'].clip(0, 1) * 20 +
            (1 - df['variability'].clip(0, 1)) * 20
        ) * 1.25
        df['quality_score'] = df['quality_score'].clip(0, 100)
 
        return df
 
    # ──────────────────────────────────────────────────────────────────
    # Aggregate to per-session feature vector
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _aggregate_features(df: pd.DataFrame) -> dict:
        def safe_mean(col): return float(df[col].mean()) if col in df.columns else 0.0
        def safe_max(col):  return float(df[col].max())  if col in df.columns else 0.0
 
        agg = {
            # Quality model features
            'avg_speed':           safe_mean('speed'),
            'peak_speed':          safe_max('speed'),
            'avg_cadence':         safe_mean('cadence'),
            'avg_stride_length':   safe_mean('stride_length'),
            'avg_asymmetry_knee':  safe_mean('asymmetry_knee'),
            'max_asymmetry_knee':  safe_max('asymmetry_knee'),
            'avg_asymmetry_stride':safe_mean('asymmetry_stride'),
            'avg_variability':     safe_mean('variability'),
            'smoothness':          safe_mean('smoothness'),
            'coordination':        safe_mean('coordination_score'),
            'efficiency':          safe_mean('efficiency_score'),
            # Coach model aliases
            'avg_stride':          safe_mean('stride_length'),
            'asymmetry':           safe_mean('asymmetry_knee'),
            'variability':         safe_mean('variability'),
            # Additional for display
            'quality_score':       safe_mean('quality_score'),
        }
        return agg
 
    # ──────────────────────────────────────────────────────────────────
    # Interpret quality score
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def interpret_quality(quality_score: float) -> dict:
        """
        Returns a dict with level, color, icon, message for a quality score (0–100).
        """
        if quality_score >= 75:
            return {
                'level': 'Excellent',
                'color': '#22c55e',
                'icon': '🏆',
                'message': 'Elite movement quality. Biomechanics are highly efficient and symmetrical.',
                'score': round(quality_score, 1),
            }
        elif quality_score >= 55:
            return {
                'level': 'Good',
                'color': '#3b82f6',
                'icon': '✅',
                'message': 'Good movement quality. Minor improvements could enhance performance.',
                'score': round(quality_score, 1),
            }
        elif quality_score >= 35:
            return {
                'level': 'Moderate',
                'color': '#f59e0b',
                'icon': '⚠️',
                'message': 'Moderate movement quality. Targeted intervention recommended.',
                'score': round(quality_score, 1),
            }
        else:
            return {
                'level': 'Low',
                'color': '#ef4444',
                'icon': '🚨',
                'message': 'Low movement quality. Significant biomechanical issues detected.',
                'score': round(quality_score, 1),
            }
 
    # ──────────────────────────────────────────────────────────────────
    # Build coaching feedback (rule-based, mirrors notebook)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def build_coaching_feedback(features: dict, quality_score: float,
                                performance_level_label: str) -> dict:
        """
        Builds structured coaching feedback from aggregated features.
 
        Args:
            features: dict from _aggregate_features()
            quality_score: predicted quality score (0–100)
            performance_level_label: 'Elite' | 'Advanced' | 'Developing'
 
        Returns:
            {
              'summary': {'level': str, 'quality': str},
              'peak_speed': str,
              'avg_cadence': float,
              'asymmetry': float,
              'consistency': float,
              'technical_focus': [str, ...],
              'recommendations': [{area, current, target, priority, drills}, ...],
              'drills': [str, ...],
              'race_strategy': [str, ...]
            }
        """
        peak_speed   = features.get('peak_speed', 0.0)
        avg_cadence  = features.get('avg_cadence', 0.0)
        avg_stride   = features.get('avg_stride_length', features.get('avg_stride', 0.0))
        asymmetry    = features.get('avg_asymmetry_knee', features.get('asymmetry', 0.0))
        variability  = features.get('avg_variability', features.get('variability', 0.0))
 
        level_desc = {
            'Elite':      'World-class performance',
            'Advanced':   'Competitive national level',
            'Developing': 'Developing — significant potential',
        }.get(performance_level_label, performance_level_label)
 
        feedback = {
            'summary':        {'level': level_desc, 'quality': f'{quality_score:.1f}/100'},
            'peak_speed':     f'{peak_speed:.2f} m/s',
            'avg_cadence':    round(avg_cadence, 3),
            'asymmetry':      round(asymmetry, 2),
            'consistency':    round(1 - min(variability, 1.0), 3),
            'technical_focus': [],
            'recommendations': [],
            'drills':          [],
            'race_strategy':   [
                'Start conservatively (95% of max first 30m)',
                'Focus on relaxation in middle 40m',
                'Accelerate through finish line',
            ],
        }
 
        # Technical flags
        if avg_cadence > 0 and avg_cadence < CADENCE_TARGET:
            feedback['technical_focus'].append(
                f'Cadence too low ({avg_cadence:.2f} Hz) → target {CADENCE_TARGET}-4.4 Hz'
            )
        if avg_stride > 0 and avg_stride < STRIDE_TARGET:
            feedback['technical_focus'].append(
                f'Stride length short ({avg_stride:.2f} m) → target {STRIDE_TARGET}-2.5 m'
            )
        if asymmetry > ASYMMETRY_TARGET:
            feedback['technical_focus'].append(
                f'High asymmetry ({asymmetry:.1f}°) → target <5°'
            )
 
        # Recommendations
        if asymmetry > 8:
            feedback['recommendations'].append({
                'area': 'Knee Asymmetry',
                'current': f'{asymmetry:.1f}°',
                'target': '<6°',
                'priority': 'HIGH',
                'drills': ['Single-leg stance exercises', 'Prosthetic alignment check'],
            })
        if variability > 0.025:
            feedback['recommendations'].append({
                'area': 'Movement Consistency',
                'current': f'{variability:.3f}',
                'target': '<0.020',
                'priority': 'MEDIUM',
                'drills': ['Metronome training', 'Rhythm drills'],
            })
        if avg_stride > 0 and avg_stride < STRIDE_TARGET:
            feedback['recommendations'].append({
                'area': 'Stride Length',
                'current': f'{avg_stride:.2f} m',
                'target': f'>= {STRIDE_TARGET} m',
                'priority': 'MEDIUM',
                'drills': ['Hip flexor stretching', 'Exaggerated stride drills'],
            })
 
        # Drills per level
        level_drills = {
            'Elite':      ['Resisted sprints', 'Over-speed training', 'Block starts', 'Video analysis'],
            'Advanced':   ['Flying 30m sprints', 'Plyometric circuit', 'Prosthetic strength training'],
            'Developing': ['Basic acceleration drills', 'Form running', 'General strength training'],
        }
        feedback['drills'] = level_drills.get(performance_level_label,
                                               level_drills['Developing'])
 
        return feedback
 