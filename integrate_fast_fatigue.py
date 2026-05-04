import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import joblib
import os
from backend.model_utils import TemporalFatigueModel, load_model, load_scaler, preprocess_sequence
from backend.preprocessors.fatigue_preprocessor import FEATURES_FATIGUE as feature_columns

MODEL_PATH = 'fast_fatigue_model.pt'
SCALER_PATH = 'backend/scalers/global_scaler.pkl'
COMPLETE_MODEL_PATH = 'fast_fatigue_model_complete.pth'
SEQ_LEN = 20
INPUT_DIM = len(feature_columns)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_complete_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH, complete_path=COMPLETE_MODEL_PATH):
    "\"\"Save existing model with scaler and metadata.\"\"\"
    model = load_model(model_path, INPUT_DIM, DEVICE)
    scaler = load_scaler(scaler_path)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': 32,
            'num_layers': 1,
            'dropout': 0.2,
            'num_heads': 2
        },
        'scaler': scaler,
        'feature_columns': feature_columns,
        'sequence_length': SEQ_LEN
    }
    
    torch.save(save_dict, complete_path)
    print(f"✅ Complete model saved to {complete_path}")
    return save_dict

def load_complete_model(filepath: str, device: torch.device = DEVICE):
    \"\"\"Load complete model package.\"\"\"
    checkpoint = torch.load(filepath, map_location=device)
    
    model = TemporalFatigueModel(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        num_layers=checkpoint['model_config']['num_layers'],
        dropout=checkpoint['model_config']['dropout'],
        num_heads=checkpoint['model_config']['num_heads']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    scaler = checkpoint['scaler']
    feature_columns = checkpoint['feature_columns']
    seq_len = checkpoint['sequence_length']
    
    print(f"✅ Loaded from {filepath} (input_dim={INPUT_DIM}, seq_len={seq_len})")
    return model, scaler, feature_columns, seq_len

def predict_fatigue(model, scaler, sequence_data: np.ndarray, device: torch.device):
    \"\"\"Predict fatigue from sequence (shape: (seq_len, n_features)).\"\"\"
    model.eval()
    
    if sequence_data.shape[0] != SEQ_LEN:
        raise ValueError(f"Expected seq_len {SEQ_LEN}, got {sequence_data.shape[0]}")
    
    # Scale
    seq_scaled = scaler.transform(sequence_data)
    seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)  # (1, seq, feats)
    
    with torch.no_grad():
        fatigue_scaled, _ = model(seq_tensor)  # Ignore quality
    
    fatigue = fatigue_scaled.item()  # Assume already inverse scaled or raw
    
    info = {
        'fatigue_raw': fatigue,
        'sequence_shape': sequence_data.shape
    }
    
    return fatigue, info

class FatiguePredictor:
    def __init__(self, model_path: str = COMPLETE_MODEL_PATH, device: torch.device = DEVICE):
        self.model, self.scaler, self.feature_columns, self.seq_len = load_complete_model(model_path, device)
        self.device = device
        print(f"✅ Predictor ready (seq_len={self.seq_len}, feats={len(self.feature_columns)})")
    
    def predict(self, sequence: np.ndarray):
        return predict_fatigue(self.model, self.scaler, sequence, self.device)

# Demo: Generate synthetic data and save complete model
def generate_sample_sequence():
    \"\"\"Generate realistic synthetic sequence matching features.\"\"\"
    seq = np.random.rand(SEQ_LEN, INPUT_DIM) * np.array([1.5, 1.5, 180, 60, 60, 60, 60, 30, 30, 0.2, 3.0, 0.1, 1.2, 0.2, 10, 8, 0.15, 8, 6, 0.5, 0.8])  # Realistic ranges
    seq[:,0] = np.linspace(0, 19, SEQ_LEN)  # time-like
    return seq

if __name__ == '__main__':
    # Save complete model from existing
    save_complete_model()
    
    # Test load + predict
    predictor = FatiguePredictor()
    sample_seq = generate_sample_sequence()
    fatigue, info = predictor.predict(sample_seq)
    print(f"Sample prediction: Fatigue = {fatigue:.4f}")
    print(info)

