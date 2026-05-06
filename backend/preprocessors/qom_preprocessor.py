"""
QoM Transformer Preprocessor
============================
Handles temporal sequence preparation for QoM Transformer model.

FEATURES (must match training):
['speed_kmh', 'cadence', 'stride_length', 'knee_left', 'knee_right', 
 'hip_left', 'hip_right', 'asymmetry_knee', 'variability', 'fatigue']

Expects sequences of length 30 (or checkpoint seq_length).
"""

import numpy as np
import pandas as pd

QOM_FEATURES = [
    'speed_kmh', 'cadence', 'stride_length', 'knee_left', 'knee_right',
    'hip_left', 'hip_right', 'asymmetry_knee', 'variability', 'fatigue'
]

QOM_SEQ_LENGTH = 30  # Default, overridden by checkpoint

class QoMPreprocessor:
    @staticmethod
    def prepare_sequence_from_df(df: pd.DataFrame, seq_length: int = QOM_SEQ_LENGTH) -> dict:
        """
        Process DataFrame → model-ready sequences (last N rows).
        
        Args:
            df: columns including QOM_FEATURES + time/runner_id (opt)
            
        Returns:
            {
                'sequence': np.array (seq_length, 10) — scaled? No, scaler separate
                'time_axis': list,
                'features': dict of aggregates,
                'rows_used': int  # last seq_length rows
            }
        """
        df = QoMPreprocessor._normalize_columns(df)
        
        # Select last seq_length rows
        if len(df) < seq_length:
            raise ValueError(f'Need >= {seq_length} rows for QoM Transformer, got {len(df)}')
        
        seq_df = df.tail(seq_length).copy()
        time_axis = seq_df['time'].tolist() if 'time' in seq_df else list(range(len(seq_df)))
        
        # Extract features (fill missing with 0)
        seq_data = []
        for _, row in seq_df.iterrows():
            row_feats = [float(row.get(f, 0.0)) for f in QOM_FEATURES]
            seq_data.append(row_feats)
        
        sequence = np.array(seq_data, dtype=np.float32)
        
        # Aggregates for display
        features = {
            'avg_speed_kmh': float(seq_df['speed_kmh'].mean()) if 'speed_kmh' in seq_df else 0.0,
            'avg_cadence': float(seq_df['cadence'].mean()) if 'cadence' in seq_df else 0.0,
            'max_asymmetry_knee': float(seq_df['asymmetry_knee'].max()) if 'asymmetry_knee' in seq_df else 0.0,
            'rows_used': len(seq_df)
        }
        
        return {
            'sequence': sequence,
            'time_axis': time_axis,
            'features': features,
            'rows_used': len(seq_df)
        }
    
    @staticmethod
    def prepare_sequence_from_manual(data: dict, seq_length: int = QOM_SEQ_LENGTH) -> np.ndarray:
        """
        Manual input → padded sequence (repeat last row).
        """
        # Extract single row features
        row_feats = [float(data.get(f, 0.0)) for f in QOM_FEATURES]
        
        # Repeat to fill sequence
        sequence = np.tile(row_feats, (seq_length, 1)).astype(np.float32)
        return sequence
    
    @staticmethod
    def interpret_qom(qom_score: float) -> dict:
        """
        Interpret QoM score (0-1) → human-readable.
        """
        if qom_score >= 0.8:
            return {
                'level': 'Excellent',
                'color': '#22c55e',
                'icon': '🏆',
                'message': 'Elite Quality of Motion. Highly efficient biomechanics.',
                'score': round(qom_score, 4),
                'percent': f'{qom_score*100:.1f}%'
            }
        elif qom_score >= 0.6:
            return {
                'level': 'Good',
                'color': '#3b82f6',
                'icon': '✅',
                'message': 'Good movement quality. Minor symmetry/efficiency gains possible.',
                'score': round(qom_score, 4),
                'percent': f'{qom_score*100:.1f}%'
            }
        elif qom_score >= 0.4:
            return {
                'level': 'Moderate',
                'color': '#f59e0b',
                'icon': '⚠️',
                'message': 'Moderate QoM. Focus on asymmetry and variability.',
                'score': round(qom_score, 4),
                'percent': f'{qom_score*100:.1f}%'
            }
        else:
            return {
                'level': 'Low',
                'color': '#ef4444',
                'icon': '🚨',
                'message': 'Low QoM detected. Urgent biomechanical intervention needed.',
                'score': round(qom_score, 4),
                'percent': f'{qom_score*100:.1f}%'
            }
    
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        # Speed kmh → ms if needed (assume kmh as per model)
        if 'speed' in df and 'speed_kmh' not in df:
            df['speed_kmh'] = df['speed']
        
        # Asymmetry from left/right
        if 'asymmetry_knee' not in df and all(c in df for c in ['knee_left', 'knee_right']):
            df['asymmetry_knee'] = abs(df['knee_left'] - df['knee_right'])
        
        return df

# For multiple windows (upload)
def predict_multiple_windows(sequence, seq_length, stride=5):
    """
    Slide window over sequence → multiple predictions.
    """
    predictions = []
    n = len(sequence) - seq_length + 1
    for i in range(0, max(0, n), stride):
        window = sequence[i:i+seq_length]
        predictions.append({
            'window_start': i,
            'window_end': i+seq_length,
            'sequence': window
        })
    return predictions
