import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from backend.preprocessors.fatigue_preprocessor import FEATURES_FATIGUE as feature_columns

# Correct model arch for fast_fatigue_model.pt (Transformer, not LSTM)
class FastTemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True, dropout=dropout)
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

MODEL_PATH = 'fast_fatigue_model.pt'
SCALER_PATH = 'backend/scalers/global_scaler.pkl'
COMPLETE_PATH = 'fast_fatigue_model_complete.pth'
SEQ_LEN = 20
INPUT_DIM = len(feature_columns)
DEVICE = torch.device('cpu')

def save_complete_model():
    model = FastTemporalTransformer(input_dim=INPUT_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {'input_dim': INPUT_DIM, 'd_model': 64, 'dropout': 0.3},
        'scaler': scaler,
        'feature_columns': feature_columns,
        'sequence_length': SEQ_LEN
    }
    
    torch.save(save_dict, COMPLETE_PATH)
    print(f"✅ Complete model saved: {COMPLETE_PATH}")

def load_complete_model(path=COMPLETE_PATH):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = FastTemporalTransformer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler = checkpoint['scaler']
    print("✅ Model loaded!")
    return model, scaler, checkpoint['feature_columns'], checkpoint['sequence_length']

def predict(model, scaler, seq):
    scaler.transform(seq)
    tensor = torch.FloatTensor(seq).unsqueeze(0)
    with torch.no_grad():
        fatigue, _ = model(tensor)
    return fatigue.item()

if __name__ == '__main__':
    print("Integrating model...")
    save_complete_model()
    
    model, scaler, feats, sl = load_complete_model()
    # Synthetic test
    seq = np.random.randn(sl, INPUT_DIM).clip(0, 2)
    pred = predict(model, scaler, seq)
    print(f"Test prediction: {pred:.4f}")
    print("Model integration complete! Test with python model_integration_complete.py")

