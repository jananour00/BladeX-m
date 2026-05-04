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


class FastTemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.fatigue_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.asym_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        fatigue = self.fatigue_head(x)
        asym = self.asym_head(x)
        return fatigue, asym


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

FAST_SEQ_LEN = 20

def load_fast_fatigue_model(model_path, device='cpu'):
    input_dim = len(FEATURES_TEMPORAL_FATIGUE)
    model = FastTemporalTransformer(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_fast_fatigue_scaler(scaler_path='backend/scalers/global_scaler.pkl'):
    return load_scaler(scaler_path)

def fast_preprocess_sequence(raw_sequence, scaler):
    """
    Preprocess for fast fatigue Transformer: expects (20, 21) or list of 20 dicts.
    """
    if isinstance(raw_sequence[0], dict):
        arr = np.array([[row.get(col, 0.0) for col in FEATURES_TEMPORAL_FATIGUE] for row in raw_sequence])
    else:
        arr = np.array(raw_sequence)
    if arr.shape[0] != FAST_SEQ_LEN:
        raise ValueError(f"Expected {FAST_SEQ_LEN} timesteps, got {arr.shape[0]}")
    arr_scaled = scaler.transform(arr)
    return torch.FloatTensor(arr_scaled).unsqueeze(0)

def predict_fast_fatigue(model, scaler, raw_sequence, device='cpu'):
    \"\"\"
    Predict fatigue from sequence.
    Returns (fatigue_score, asymmetry_score)
    \"\"\"
    model.eval()
    tensor = fast_preprocess_sequence(raw_sequence, scaler).to(device)
    with torch.no_grad():
        fatigue, asym = model(tensor)
    return fatigue.item(), asym.item()

# ════════════════════════════════════════════════════════════════════════════
# QoM TRANSFORMER MODEL
# ════════════════════════════════════════════════════════════════════════════

import torch.nn.functional as F
from backend.preprocessors.qom_preprocessor import QOM_FEATURES, QOM_SEQ_LENGTH

class SimpleTemporalTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze()

def load_qom_model(model_path='qom_transformer_model.pth', device='cpu'):
    \"\"\"
    Load QoM Transformer model + metadata.
    \"\"\"
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SimpleTemporalTransformer(
        input_dim=checkpoint['input_dim'],
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint['feature_cols'], checkpoint['sequence_length']

def qom_preprocess_sequence(raw_sequence, scaler, feature_cols, seq_length):
    \"\"\"
    raw_sequence: df or list of dicts → (1, seq_length, 10) tensor
    \"\"\"
    if isinstance(raw_sequence, pd.DataFrame):
        seq_data = [[row.get(col, 0.0) for col in feature_cols] for _, row in raw_sequence.tail(seq_length).iterrows()]
    else:
        seq_data = [[row.get(col, 0.0) for col in feature_cols] for row in raw_sequence[-seq_length:]]
    
    arr = np.array(seq_data, dtype=np.float32)
    arr_scaled = scaler.transform(arr)
    return torch.FloatTensor(arr_scaled).unsqueeze(0)

def predict_qom(model, scaler, feature_cols, seq_length, raw_sequence, device='cpu'):
    \"\"\"
    Full QoM prediction pipeline.
    \"\"\"
    model.eval()
    tensor = qom_preprocess_sequence(raw_sequence, scaler, feature_cols, seq_length).to(device)
    
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()
    
    return float(pred[0])
    """
    Predict fatigue from sequence.
    Returns (fatigue_score, asymmetry_score)
    """
    model.eval()
    tensor = fast_preprocess_sequence(raw_sequence, scaler).to(device)
    with torch.no_grad():
        fatigue, asym = model(tensor)
    return fatigue.item(), asym.item()


