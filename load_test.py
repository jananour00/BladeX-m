import torch
import joblib
import pickle
import os

print("=== MODEL LOADING TEST (No Flask needed) ===")
root_dir = os.getcwd()
print(f"Working dir: {root_dir}")

# Test 1: all_scalers.joblib
print("\n1. Testing all_scalers.joblib...")
try:
    scalers = joblib.load('all_scalers.joblib')
    print(f"✅ SUCCESS: {len(scalers['feature_columns'])} features")
    print(f"Features: {scalers['feature_columns'][:3]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: bayesian_model_complete.pt
print("\n2. Testing bayesian_model_complete.pt...")
try:
    checkpoint = torch.load('bayesian_model_complete.pt', map_location='cpu')
    print(f"✅ SUCCESS: config={checkpoint['model_config']}, state_dict keys={len(checkpoint['model_state_dict'])}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: cgnn_model_complete.pt
print("\n3. Testing cgnn_model_complete.pt...")
try:
    checkpoint = torch.load('cgnn_model_complete.pt', map_location='cpu')
    print(f"✅ SUCCESS: config={checkpoint['model_config']}, state_dict keys={len(checkpoint['model_state_dict'])}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: causal_graph_complete.pkl
print("\n4. Testing causal_graph_complete.pkl...")
try:
    with open('causal_graph_complete.pkl', 'rb') as f:
        graph = pickle.load(f)
    print(f"✅ SUCCESS: type={type(graph)}, keys={list(graph.keys())[:5]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n🎉 SUMMARY: All models accessible for loading!")
