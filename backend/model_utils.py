import torch
import torch.nn as nn
import joblib
import os
import logging
import pickle
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

# Import preprocessors for feature info
try:
    from backend.preprocessors.cgnn_preprocessor import load_cgnn_scalers, get_cgnn_feature_columns
    from backend.preprocessors.qom_preprocessor import QOM_FEATURES
    from backend.preprocessors.temporal_fatigue_preprocessor import FEATURES_TEMPORAL_FATIGUE, SEQ_LEN
    from backend.preprocessors.bayes_preprocessor import load_scalers
except ImportError as e:
    log.warning(f"Preprocessor import failed: {e}")
    load_cgnn_scalers = lambda: None
    get_cgnn_feature_columns = lambda: []
    QOM_FEATURES = []
    FEATURES_TEMPORAL_FATIGUE = []
    SEQ_LEN = 30
    load_scalers = lambda: None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_qom_model(model_path, device):
    """
    Load QoM Transformer model + scaler.
    Returns: model, scaler, feature_cols, seq_length
    """

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        scaler_path = model_path.replace('qom_transformer_model.pth', 'qom_scaler.joblib')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        feature_cols = QOM_FEATURES
        seq_length = getattr(model, 'seq_length', 30)
        log.info(f"✅ QoM model loaded: seq_len={seq_length}, feats={len(feature_cols)}")
        return model, scaler, feature_cols, seq_length
    except Exception as e:
        log.warning(f"QoM load failed: {e}. Using dummy.")
        class DummyQoM(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.zeros(x.size(0), 1, device=device)

        return DummyQoM().to(device), None, QOM_FEATURES or ['speed_kmh'], 30

def load_cgnn_model(model_path, device):
    """
    Load CGNN model.
    Returns: model, scaler, feature_cols
    """

    try:
        load_cgnn_scalers()
        feature_cols = get_cgnn_feature_columns()
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        log.info(f"✅ CGNN model loaded: feats={len(feature_cols)}")
        return model, None, feature_cols
    except Exception as e:
        log.warning(f"CGNN load failed: {e}. Using dummy.")
        class DummyCGNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                b = x.size(0)
                return {'fatigue': torch.zeros(b), 'qom': torch.zeros(b), 'injury_risk': torch.zeros(b)}

        return DummyCGNN().to(device), None, []

def load_temporal_model(model_path, n_features, device):
    """
    Load temporal fast fatigue model.
    """

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        log.info(f"✅ Temporal model loaded: feats={n_features}")
        return model
    except Exception as e:
        log.warning(f"Temporal load failed: {e}. Using dummy.")
        class DummyTemporal(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                b, seq, f = x.shape
                return torch.zeros(b, seq, device=device)

        return DummyTemporal().to(device)

def load_temporal_scaler(scaler_path):
    try:
        return joblib.load(scaler_path)
    except:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()

def predict_qom(model, scaler, feature_cols, seq_length, data_df, device):
    """
    Helper for app.py QoM prediction.
    """

    from backend.preprocessors.qom_preprocessor import QoMPreprocessor
    sequence = QoMPreprocessor.prepare_sequence_from_df(data_df, seq_length)['sequence']
    if scaler:
        sequence = scaler.transform(sequence.reshape(-1, len(feature_cols)).reshape(sequence.shape))
    sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(sequence).squeeze(-1).cpu().numpy()
    return float(pred.mean())

def predict_cgnn(model, X_scaled, device):
    X = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        outputs = model(X)
    return outputs

