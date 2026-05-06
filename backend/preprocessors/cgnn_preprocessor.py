"""
CGNN Preprocessor
Handles feature preparation for CGNN model (shares scaler with Bayesian).
Expected in root: all_scalers.joblib, cgnn_model_complete.pt
"""

import os
import numpy as np
import joblib
import torch
from typing import Dict, Any, List

# Reuse Bayesian scalers (same 11 features)
from .bayes_preprocessor import (
    load_scalers,
    get_feature_columns,
    _scalers as _bayes_scalers,
    _feature_columns as _bayes_feature_columns
)

def load_cgnn_scalers():
    """Load shared scalers for CGNN (same as Bayesian)."""
    load_scalers()  # Ensures _bayes_scalers loaded

def get_cgnn_feature_columns() -> List[str]:
    """CGNN features (identical to Bayesian)."""
    if _bayes_feature_columns is None:
        load_cgnn_scalers()
    return _bayes_feature_columns

def prepare_input(features: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare features dict for CGNN prediction (identical to Bayesian).
    
    Args:
        features: dict with keys in get_cgnn_feature_columns()
        
    Returns:
        {
            'X_scaled': np.array (1, 11) - scaled features,
            'feature_vector': list - original values,
            'computed': dict - derived metrics (asymmetry etc.)
        }
    """
    load_cgnn_scalers()
    
    feature_vector = []
    computed = {}
    
    for col in get_cgnn_feature_columns():
        if col not in features:
            raise ValueError(f'Missing CGNN feature: {col}')
        val = float(features[col])
        feature_vector.append(val)
    
    # Scale using shared scaler
    X_scaled = _bayes_scalers['scaler_X'].transform([feature_vector])
    
    # Computed metrics (for UI display)
    knee_asym = abs(features.get('knee_left', 0) - features.get('knee_right', 0))
    computed['knee_asymmetry_deg'] = round(knee_asym, 2)
    computed['hip_asymmetry_deg'] = round(abs(features.get('hip_left', 0) - features.get('hip_right', 0)), 2)
    
    return {
        'X_scaled': X_scaled,
        'feature_vector': feature_vector,
        'computed': computed
    }

def interpret_cgnn_predictions(outputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """
    Format CGNN predictions (fatigue, qom, injury_risk).
    Assumes model outputs dict or tensor[3].
    """
    # Handle both dict and tensor outputs
    if isinstance(outputs, dict):
        fatigue = float(outputs['fatigue'])
        qom = float(outputs['qom'])
        injury_risk = float(outputs['injury_risk'])
    else:  # Assume tensor[3]
        fatigue, qom, injury_risk = outputs.cpu().numpy()
    
    result = {}
    
    # Fatigue
    level_f = 'Low' if fatigue < 0.4 else 'Moderate' if fatigue < 0.7 else 'High'
    color_f = '#34d399' if fatigue < 0.4 else '#f59e0b' if fatigue < 0.7 else '#f87171'
    result['fatigue'] = {
        'value': round(fatigue, 4),
        'level': level_f,
        'color': color_f,
        'risk_message': f"{level_f} fatigue risk ({fatigue*100:.1f}%)"
    }
    
    # QoM
    level_q = 'Excellent' if qom > 0.8 else 'Good' if qom > 0.6 else 'Fair' if qom > 0.4 else 'Poor'
    color_q = '#10b981' if qom > 0.8 else '#3b82f6' if qom > 0.6 else '#f59e0b' if qom > 0.4 else '#f87171'
    result['qom'] = {
        'value': round(qom, 4),
        'level': level_q,
        'color': color_q,
        'score_pct': round(qom * 100, 1)
    }
    
    # Injury Risk
    level_i = 'Low' if injury_risk < 0.3 else 'Moderate' if injury_risk < 0.6 else 'High'
    color_i = '#34d399' if injury_risk < 0.3 else '#f59e0b' if injury_risk < 0.6 else '#f87171'
    result['injury_risk'] = {
        'value': round(injury_risk, 4),
        'level': level_i,
        'color': color_i,
        'risk_message': f"{level_i} injury risk ({injury_risk*100:.1f}%)"
    }
    
    return result

# Auto-load on import
try:
    load_cgnn_scalers()
except:
    pass

