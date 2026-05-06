import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.preprocessors.fatigue_preprocessor import FatiguePreprocessor, FEATURES_INJURY
import joblib

inj = joblib.load('models/fatigue_injury_model/injury_pipeline.joblib')
scaler = inj.named_steps['scaler']

# Check training stats for injury features
print("=== Injury Model Training Stats ===")
for i, f in enumerate(FEATURES_INJURY):
    print(f"  {f:25s} mean={scaler.mean_[i]:8.4f}  std={scaler.scale_[i]:8.4f}")

print("\n=== Speed vs Injury Probability ===")
for spd in [3, 5, 8, 10, 12]:
    data = {
        'speed': spd, 'stride_length': spd*0.25, 'cadence': spd*18,
        'knee_left': 45, 'knee_right': 42, 'hip_left': 20, 'hip_right': 18,
        'weight_kg': 72, 'height_cm': 178, 'peak_speed_ms': spd * 0.95,
        'variability': 0.02 + spd * 0.003,
        'prosthetic_side': 'right', 'asymmetry_stride': 0.05 + spd * 0.005,
    }
    p = FatiguePreprocessor.prepare_input(data)
    proba = inj.predict_proba(p['X_injury'])[0]
    print(f"  Speed={spd:5.1f} m/s => P(injury)={proba[1]:.4f}")

# Test with extreme asymmetry
print("\n=== High Asymmetry Scenario ===")
for asym in [0.05, 0.10, 0.20, 0.40]:
    data = {
        'speed': 12, 'stride_length': 3.0, 'cadence': 216,
        'knee_left': 60, 'knee_right': 30, 'hip_left': 25, 'hip_right': 10,
        'weight_kg': 72, 'height_cm': 178, 'peak_speed_ms': 12,
        'variability': 0.06,
        'prosthetic_side': 'right', 'asymmetry_stride': asym,
    }
    p = FatiguePreprocessor.prepare_input(data)
    proba = inj.predict_proba(p['X_injury'])[0]
    print(f"  asym_stride={asym:.2f} => P(injury)={proba[1]:.4f}")
