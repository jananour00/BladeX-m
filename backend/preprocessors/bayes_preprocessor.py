"""
Bayesian Biomechanics Preprocessor
Handles feature preparation for Bayesian Neural Network model.
Expected in root: all_scalers.joblib
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, List

# Load scalers and features (singleton-like)
_scalers = None
_feature_columns = None

def load_scalers():
    global _scalers, _feature_columns
    scaler_path = os.path.join(os.path.dirname(__file__), '..', '..', 'all_scalers.joblib')
    if os.path.exists(scaler_path):
        _scalers = joblib.load(scaler_path)
        _feature_columns = _scalers['feature_columns']
        _scalers['scaler_X']  # Ensure exists
    else:
        raise FileNotFoundError(f"all_scalers.joblib not found at {scaler_path}")

def get_feature_columns() -> List[str]:
    if _feature_columns is None:
        load_scalers()
    return _feature_columns

FEATURES_BAYES = [
    'speed', 'speed_kmh', 'stride_length', 'cadence',
    'knee_left', 'knee_right', 'hip_left', 'hip_right',
    'asymmetry_knee', 'asymmetry_stride', 'variability'
]

def prepare_input(features: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare features dict for Bayesian model prediction.
    
    Args:
        features: dict with keys in FEATURES_BAYES
        
    Returns:
        {
            'X_scaled': np.array (1, 11) - scaled features,
            'feature_vector': list - original values,
            'computed': dict - derived metrics
        }
    """
    if _scalers is None:
        load_scalers()
    
    feature_vector = []
    computed = {}
    
    for col in _feature_columns:
        if col not in features:
            raise ValueError(f'Missing feature: {col}')
        val = float(features[col])
        feature_vector.append(val)
    
    # Scale
    X_scaled = _scalers['scaler_X'].transform([feature_vector])
    
    # Computed metrics (for display)
    knee_asym = abs(features.get('knee_left', 0) - features.get('knee_right', 0))
    computed['knee_asymmetry_deg'] = round(knee_asym, 2)
    computed['hip_asymmetry_deg'] = round(abs(features.get('hip_left', 0) - features.get('hip_right', 0)), 2)
    
    return {
        'X_scaled': X_scaled,
        'feature_vector': feature_vector,
        'computed': computed
    }

def interpret_predictions(predictions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Format Bayesian predictions for API response.
    """
    result = {}
    for key, pred in predictions.items():
        mean = pred['mean']
        std = pred['std']
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std
        
        level = 'Low'
        color = '#34d399'
        if key == 'fatigue':
            if mean > 0.7: level = 'High'; color = '#f87171'
            elif mean > 0.4: level = 'Moderate'; color = '#f59e0b'
        elif key == 'injury_risk':
            if mean > 0.5: level = 'High'; color = '#f87171'
            elif mean > 0.2: level = 'Moderate'; color = '#f59e0b'
        
        result[key] = {
            'value': round(mean, 4),
            'uncertainty': round(std, 4),
            'confidence_interval': [round(ci_low, 4), round(ci_high, 4)],
            'level': level,
            'color': color,
            'risk_message': f"{level} {key.replace('_', ' ').title()} ({std*100:.1f}% uncertainty)"
        }
    
    return result

# Auto-load on import
try:
    load_scalers()
except:
    pass

