import torch
import torch.nn as nn
import numpy as np
import pickle
from backend.preprocessors.fatigue_preprocessor import FEATURES_FATIGUE as FEATURES_TEMPORAL_FATIGUE

# Model architecture (matches training)
class TemporalFatigueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2, num_heads=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads,
                                               batch_first=True, dropout=dropout)
        self.fatigue_head = nn.Sequential(
            nn.Linear(hidden_dim*2, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim*2, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)
        fatigue = self.fatigue_head(pooled)
        quality = self.quality_head(pooled)
        return fatigue, quality

def load_model(model_path, input_dim, device='cpu'):
    model = TemporalFatigueModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_scaler(scaler_path='backend/scalers/global_scaler.pkl'):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def preprocess_sequence(raw_sequence, scaler, feature_cols_order):
    """
    raw_sequence: list of 20 dicts 
    scaler: fitted StandardScaler
    feature_cols_order: FEATURES_TEMPORAL_FATIGUE (21 features)
    """
    if isinstance(raw_sequence[0], dict):
        arr = np.array([[row.get(col, 0.0) for col in feature_cols_order] for row in raw_sequence])
    else:
        arr = np.array(raw_sequence)
    arr_scaled = scaler.transform(arr)
    tensor = torch.FloatTensor(arr_scaled).unsqueeze(0)  # (1, 20, 21)
    return tensor

