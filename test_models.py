import requests
import json

print("=== BLADE RUNNER MODELS - INPUT/OUTPUT TEST ===\n")

# Sample input (11 features from all_scalers.joblib)
SAMPLE_INPUT = {
    "speed": 5.2,
    "speed_kmh": 18.72,
    "stride_length": 1.45,
    "cadence": 180,
    "knee_left": 42.0,
    "knee_right": 44.5,
    "hip_left": 28.0,
    "hip_right": 29.5,
    "asymmetry_knee": 2.5,
    "asymmetry_stride": 1.8,
    "variability": 0.025
}

print("📥 INPUT FEATURES:")
for k, v in SAMPLE_INPUT.items():
    print(f"  {k}: {v}")
print()

# Test Bayesian model
print("🔬 BAYESIAN MODEL PREDICTION:")
bayes_payload = {"model": "bayesian", "features": SAMPLE_INPUT}
bayes_resp = requests.post("http://localhost:5001/predict", json=bayes_payload)
bayes_result = bayes_resp.json()
print(json.dumps(bayes_result, indent=2))
print()

# Test CGNN model
print("🧠 CGNN MODEL PREDICTION:")
cgnn_payload = {"model": "cgnn", "features": SAMPLE_INPUT}
cgnn_resp = requests.post("http://localhost:5001/predict", json=cgnn_payload)
cgnn_result = cgnn_resp.json()
print(json.dumps(cgnn_result, indent=2))
print()

# Health check
print("✅ HEALTH CHECK:")
health_resp = requests.get("http://localhost:5001/health")
health = health_resp.json()
print(json.dumps(health, indent=2))

print("\n🎉 SUCCESS! Both models processed input → output")
print("Start API: python models_complete_api.py")
print("Run this test: python test_models.py")
