"""
Temporal Fatigue Preprocessor
=============================
Prepares time-series data for fast_fatigue_model.pt (temporal transformer).

Assumes input features match global_scaler.pkl (fatigue-related).
Uses rolling windows for sequence preparation.

Expected CSV columns: time, speed, stride_length, cadence, knee_left, knee_right, 
hip_left, hip_right, variability, asymmetry_stride, prosthetic_side, etc.
"""
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any

from .fatigue_preprocessor import FatiguePreprocessor, FEATURES_FATIGUE

# Temporal-specific sequence config
SEQ_LEN = 50
STRIDE = 10
FEATURES_TEMPORAL_FATIGUE = FEATURES_FATIGUE[:12]  # Subset for temporal model

class TemporalFatiguePreprocessor:
    @staticmethod
    def prepare_sequence(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> Dict[str, Any]:
        """
        Process DataFrame → scaled sequences for temporal model.
        
        Steps:
        1. Normalize columns
        2. Parse CSV → list of engineered feature dicts
        3. Build DataFrame with temporal features
        4. Scale with global_scaler
        5. Create sliding windows
        """
        # Normalize columns
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        # Engineer features using fatigue preprocessor
        rows = FatiguePreprocessor.parse_csv(df)
        
        if not rows:
            raise ValueError("No valid rows after parsing")
        
        # Build DataFrame from engineered rows
        df_engineered = pd.DataFrame(rows)
        
        # Select temporal features (handle missing gracefully)
        available_features = [f for f in FEATURES_TEMPORAL_FATIGUE if f in df_engineered.columns]
        if len(available_features) < 6:  # Minimum sensible features
            raise ValueError(f"Insufficient temporal features found: {available_features}")
        
        X = df_engineered[available_features].fillna(0).values
        
        # Load scaler
        scaler_path = '../scalers/global_scaler.pkl'
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        
        # Create sliding windows
        sequences = []
        time_axis = []
        for i in range(0, len(X_scaled) - seq_len + 1, STRIDE):
            sequences.append(X_scaled[i:i+seq_len])
            time_axis.append(df.index[i] if hasattr(df, 'index') else i)
        
        if not sequences:
            raise ValueError("No sequences could be generated (need at least seq_len rows)")
        
        return {
            'sequences': np.array(sequences),
            'time_axis': time_axis,
            'scaler': scaler,
            'features': available_features,
            'n_features': len(available_features),
            'seq_len': seq_len
        }
    
    @staticmethod
    def prepare_single_sequence(data: Dict[str, Any], seq_len: int = SEQ_LEN) -> np.ndarray:
        """
        Single manual input → padded sequence.
        """
        # Engineer single row
        engineered = FatiguePreprocessor._engineer_features(data)
        
        # Select features
        available_features = [f for f in FEATURES_TEMPORAL_FATIGUE if f in engineered]
        feature_values = [float(engineered.get(f, 0.0)) for f in available_features]
        
        # Load scaler
        scaler_path = '../scalers/global_scaler.pkl'
        scaler = joblib.load(scaler_path)
        scaled_values = scaler.transform([feature_values])[0]
        
        # Pad to seq_len with zeros (no history)
        sequence = np.zeros((seq_len, len(available_features)))
        sequence[0] = scaled_values  # Current observation at t=0
        
        return sequence[np.newaxis, ...]  # Add batch dim: (1, seq_len, n_features)
    
    @staticmethod
    def interpret_predictions(predictions: np.ndarray) -> Dict[str, Any]:
        """
        Interpret temporal model outputs → risk levels/time-series.
        """
        fatigue_series = predictions.squeeze()
        
        max_fatigue = float(np.max(fatigue_series))
        avg_fatigue = float(np.mean(fatigue_series))
        trend = 'increasing' if fatigue_series[-5:].mean() > fatigue_series[:-5].mean() else 'stable'
        
        return {
            'fatigue_series': fatigue_series.tolist(),
            'max_fatigue': round(max_fatigue, 4),
            'avg_fatigue': round(avg_fatigue, 4),
            'trend': trend,
            'risk_level': 'High' if max_fatigue > 0.8 else 'Moderate' if max_fatigue > 0.6 else 'Low',
            'recommendation': 'Rest recommended' if max_fatigue > 0.8 else 'Monitor'
        }

