import torch
import joblib
import pickle
import os
import numpy as np

print("=== TERMINAL MODEL TEST - INPUT/OUTPUT LIVE ===")
print("PyTorch:", torch.__version__)

# Architectures (copied from your app.py)
class BayesianBiomechanicsModel(torch.nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64):
        super().__init__()
        # Simplified for test - use your exact architecture
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 3)  # fatigue, qom, injury
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return {'fatigue': out[:,0], 'qom': torch.sigmoid(out[:,1]), 'injury_risk': torch.sigmoid(out[:,2])}

class SimpleCGNN(torch.nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, output_dim=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return {'fatigue': out[:,0], 'qom': torch.sigmoid(out[:,1]), 'injury_risk': torch.sigmoid(out[:,2])}

device = torch.device('cpu')

# SAMPLE INPUT (11 features)
SAMPLE_INPUT = np.array([5.2, 18.72, 1.45, 180, 42.0, 44.5, 28.0, 29.5, 2.5, 1.8, 0.025], dtype=np.float32)
print("\n📥 INPUT (11 features):")
print(SAMPLE_INPUT)
print()

try:
    # 1. SCALERS
    print("1. all_scalers.joblib:")
    scalers = joblib.load('all_scalers.joblib')
    scaler_X = scalers['scaler_X']
    features = scalers['feature_columns']
    print(f"Features: {features}")
    X_scaled = scaler_X.transform([SAMPLE_INPUT])
    print(f"Scaled: {X_scaled[0][:3]}...")
    
    # 2. BAYESIAN
    print("\n2. bayesian_model_complete.pt:")
    checkpoint = torch.load('bayesian_model_complete.pt', map_location=device, weights_only=False)
    print(f"Config keys: {list(checkpoint['model_config'].keys())}")
    print(f"Input dim: {checkpoint['model_config']['input_dim']}")
    
    model_bayes = BayesianBiomechanicsModel(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dim=checkpoint['model_config'].get('hidden_dim', 64)
    ).to(device)
    model_bayes.load_state_dict(checkpoint['model_state_dict'])
    model_bayes.eval()
    
    pred_bayes = model_bayes(torch.FloatTensor(X_scaled))
    print(f"Fatigue: {pred_bayes['fatigue'][0]:.3f}")
    print(f"QoM: {pred_bayes['qom'][0]:.3f}")
    print(f"Injury risk: {pred_bayes['injury_risk'][0]:.3f}")
    
    # 3. CGNN
    print("\n3. cgnn_model_complete.pt:")
    checkpoint_cgnn = torch.load('cgnn_model_complete.pt', map_location=device, weights_only=False)
    print(f"Config: input_dim={checkpoint_cgnn['model_config']['input_dim']}, output_dim={checkpoint_cgnn['model_config']['output_dim']}")
    
    model_cgnn = SimpleCGNN(
        input_dim=checkpoint_cgnn['model_config']['input_dim'],
        hidden_dim=checkpoint_cgnn['model_config'].get('hidden_dim', 64),
        output_dim=checkpoint_cgnn['model_config']['output_dim']
    ).to(device)
    model_cgnn.load_state_dict(checkpoint_cgnn['model_state_dict'])
    model_cgnn.eval()
    
    pred_cgnn = model_cgnn(torch.FloatTensor(X_scaled))
    print(f"Fatigue: {pred_cgnn['fatigue'][0]:.3f}")
    print(f"QoM: {pred_cgnn['qom'][0]:.3f}")
    print(f"Injury risk: {pred_cgnn['injury_risk'][0]:.3f}")
    
    # 4. CAUSAL
    print("\n4. causal_graph_complete.pkl:")
    graph = pickle.load(open('causal_graph_complete.pkl', 'rb'))
    print(f"Keys: {list(graph.keys())}")
    
    print("\n🎉 LIVE PREDICTIONS FROM YOUR MODELS!")
    print("Run: python model_test_fixed.py")

except Exception as e:
    print(f"ERROR: {e}")
    print("Files exist but loading issue - check PyTorch version/sklearn compatibility")
