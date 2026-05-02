import torch
import joblib
import pickle
import os
from models_complete_api import BayesianBiomechanicsModel, SimpleCGNN  # Import architectures

print("=== FULL MODEL INTEGRATION TEST ===")
root_dir = os.getcwd()
print(f"Root: {root_dir}")

device = torch.device('cpu')

# 1. SCALERS
print("\n1️⃣ all_scalers.joblib")
scalers = joblib.load('all_scalers.joblib')
scaler_X = scalers['scaler_X']
feature_columns = scalers['feature_columns']
print(f"✅ 11 features: {feature_columns}")

# Sample input
SAMPLE = {col: 1.0 for col in feature_columns[:11]}
print(f"Sample input keys: {list(SAMPLE.keys())}")

# Scale
X = scaler_X.transform([list(SAMPLE.values())])
print(f"Scaled shape: {X.shape}")

# 2. BAYESIAN
print("\n2️⃣ bayesian_model_complete.pt")
checkpoint_bayes = torch.load('bayesian_model_complete.pt', map_location=device, weights_only=False)
print(f"Config: {checkpoint_bayes['model_config']}")

model_bayes = BayesianBiomechanicsModel(
    input_dim=checkpoint_bayes['model_config']['input_dim'],
    hidden_dim=checkpoint_bayes['model_config']['hidden_dim']
).to(device)
model_bayes.load_state_dict(checkpoint_bayes['model_state_dict'])
model_bayes.eval()

pred_bayes = model_bayes(torch.FloatTensor(X))
print(f"✅ Prediction: fatigue={pred_bayes['fatigue']['mean']:.3f}")

# 3. CGNN
print("\n3️⃣ cgnn_model_complete.pt")
checkpoint_cgnn = torch.load('cgnn_model_complete.pt', map_location=device, weights_only=False)
print(f"Config: {checkpoint_cgnn['model_config']}")

model_cgnn = SimpleCGNN(
    input_dim=checkpoint_cgnn['model_config']['input_dim'],
    hidden_dim=checkpoint_cgnn['model_config']['hidden_dim'],
    output_dim=checkpoint_cgnn['model_config']['output_dim']
).to(device)
model_cgnn.load_state_dict(checkpoint_cgnn['model_state_dict'])
model_cgnn.eval()

pred_cgnn = model_cgnn(torch.FloatTensor(X))
print(f"✅ Prediction: qom={pred_cgnn['qom'][0]:.3f}")

# 4. CAUSAL GRAPH
print("\n4️⃣ causal_graph_complete.pkl")
graph = pickle.load(open('causal_graph_complete.pkl', 'rb'))
print(f"✅ Keys: {list(graph.keys())}")

print("\n🎉 ALL 4 MODELS FULLY LOADED & TESTED!")
print("\nFIX API: Use exact config extraction like above.")
