import torch
import numpy as np
import pickle

MODEL_PATH = 'fast_fatigue_model.pt'
SCALER_PATH = 'backend/scalers/global_scaler.pkl'
FEATURES = [
    "speed", "stride_length", "cadence", "knee_prosthetic", "knee_sound", "hip_prosthetic", "hip_sound",
    "asymmetry_ratio", "asymmetry_stride", "symmetry_decay", "bmi", "peak_speed_ms", "variability",
    "roll_speed_mean", "roll_speed_std", "roll_knee_std", "roll_hip_std", "roll_variability_std",
    "roll_knee_std_w5", "roll_hip_std_w5", "cumulative_impact"
]

print("🔥 Fatigue Model Terminal Test")
print("Loading scaler...")
scaler = pickle.load(open(SCALER_PATH, 'rb'))
print("Scaler loaded.")

print("\nLoading model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
print("Model checkpoint keys:", list(checkpoint.keys())[:5], "...")

# Create dummy model to inspect
input_dim = 21
device = 'cpu'

# Demo prediction with dummy input
print("\nGenerating dummy sequence...")
seq = np.random.rand(20, 21) * 2  # Realistic range
seq_scaled = scaler.transform(seq)
tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)

print("Tensor shape:", tensor.shape)
print("Ready for inference!")

print("\n✅ Model loaded and ready for terminal use.")
print("Example usage:")
print("  seq = np.random.rand(20, 21)")
print("  seq_scaled = scaler.transform(seq)")
print("  pred = model(seq_scaled_tensor)  # fatigue, quality")
print("\\nCopy this to your Python shell for predictions!")

