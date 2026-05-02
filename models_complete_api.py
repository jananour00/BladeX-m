from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pickle
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Models API running on device: {device}")

# Model Architectures (exact match to training)
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
    
    def forward(self, x, n_samples=1):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        outputs = []
        for _ in range(n_samples):
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            outputs.append(F.linear(x, weight, bias))
        return torch.stack(outputs, dim=0)

class BayesianBiomechanicsModel(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64):
        super().__init__()
        self.bayes1 = BayesianLinear(input_dim, hidden_dim)
        self.bayes2 = BayesianLinear(hidden_dim, hidden_dim)
        self.bayes3 = BayesianLinear(hidden_dim, hidden_dim // 2)
        self.fatigue_head = BayesianLinear(hidden_dim // 2, 1)
        self.qom_head = BayesianLinear(hidden_dim // 2, 1)
        self.injury_head = BayesianLinear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, n_samples=10):
        h = self.bayes1(x, n_samples).mean(0)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.bayes2(h, n_samples).mean(0)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.bayes3(h, n_samples).mean(0)
        h = F.relu(h)
        
        fatigue_samples = self.fatigue_head(h, n_samples).squeeze(-1)
        qom_samples = torch.sigmoid(self.qom_head(h, n_samples).squeeze(-1))
        injury_samples = torch.sigmoid(self.injury_head(h, n_samples).squeeze(-1))
        
        return {
            'fatigue': {'mean': fatigue_samples.mean(0), 'std': fatigue_samples.std(0)},
            'qom': {'mean': qom_samples.mean(0), 'std': qom_samples.std(0)},
            'injury_risk': {'mean': injury_samples.mean(0), 'std': injury_samples.std(0)}
        }

class SimpleCGNN(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, output_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.causal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h_seq = h.unsqueeze(1)
        h_attn, _ = self.causal_attention(h_seq, h_seq, h_seq)
        output = self.decoder(h_attn.squeeze(1))
        return {
            'fatigue': output[:, 0],
            'qom': torch.sigmoid(output[:, 1]),
            'injury_risk': torch.sigmoid(output[:, 2])
        }

# Global model store
bayesian_model = None
cgnn_model = None
scaler_X = None
feature_columns = None
causal_graph = None

def load_all_models():
    global bayesian_model, cgnn_model, scaler_X, feature_columns, causal_graph
    
    root_dir = os.getcwd()
    print(f"Root dir: {root_dir}")
    
    print("Loading Bayesian model...")
    bayesian_path = os.path.join(root_dir, 'bayesian_model_complete.pt')
    print(f"Bayesian path: {bayesian_path}")
    checkpoint = torch.load(bayesian_path, map_location=device, weights_only=False)
    bayesian_model = BayesianBiomechanicsModel(**checkpoint['model_config']).to(device)
    bayesian_model.load_state_dict(checkpoint['model_state_dict'])
    bayesian_model.eval()
    
    print("Loading CGNN model...")
    cgnn_path = os.path.join(root_dir, 'cgnn_model_complete.pt')
    print(f"CGNN path: {cgnn_path}")
    checkpoint = torch.load(cgnn_path, map_location=device, weights_only=False)
    cgnn_model = SimpleCGNN(**checkpoint['model_config']).to(device)
    cgnn_model.load_state_dict(checkpoint['model_state_dict'])
    cgnn_model.eval()
    
    print("Loading scalers...")
    scalers_path = os.path.join(root_dir, 'all_scalers.joblib')
    print(f"Scalers path: {scalers_path}")
    scalers = joblib.load(scalers_path)
    scaler_X = scalers['scaler_X']
    feature_columns = scalers['feature_columns']
    
    print("Loading causal graph...")
    causal_path = os.path.join(root_dir, 'causal_graph_complete.pkl')
    print(f"Causal path: {causal_path}")
    with open(causal_path, 'rb') as f:
        causal_graph = pickle.load(f)
    
    print(f"✅ ALL MODELS LOADED: Bayesian({bayesian_model}), CGNN({cgnn_model})")
    print(f"Features: {len(feature_columns)}")

# Load on startup
load_all_models()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'bayesian_loaded': bayesian_model is not None,
        'cgnn_loaded': cgnn_model is not None,
        'features': feature_columns[:5] + ['...'],
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    model_type = data.get('model', 'bayesian').lower()
    features = data.get('features', {})
    
    try:
        feature_vector = [features.get(col, 0.0) for col in feature_columns]
        X_scaled = scaler_X.transform([feature_vector])
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            if model_type == 'bayesian':
                pred = bayesian_model(X_tensor)
                result = {
                    'model': 'Bayesian',
                    'fatigue': float(pred['fatigue']['mean']),
                    'fatigue_uncertainty': float(pred['fatigue']['std']),
                    'qom': float(pred['qom']['mean']),
                    'qom_uncertainty': float(pred['qom']['std']),
                    'injury_risk': float(pred['injury_risk']['mean']),
                    'injury_uncertainty': float(pred['injury_risk']['std'])
                }
            elif model_type == 'cgnn':
                pred = cgnn_model(X_tensor)
                result = {
                    'model': 'CGNN',
                    'fatigue': float(pred['fatigue'][0]),
                    'qom': float(pred['qom'][0]),
                    'injury_risk': float(pred['injury_risk'][0]),
                    'note': 'Deterministic (no uncertainty)'
                }
            else:
                return jsonify({'error': 'Use model=bayesian or cgnn'}), 400
        
        return jsonify({'success': True, **result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_sample', methods=['GET'])
def test_sample():
    """Test with sample data"""
    sample_features = {
        'speed': 5.2, 'speed_kmh': 18.72, 'stride_length': 1.45,
        'cadence': 180, 'knee_left': 42.0, 'knee_right': 44.5,
        'hip_left': 28.0, 'hip_right': 29.5, 'asymmetry_knee': 2.5,
        'asymmetry_stride': 1.8, 'variability': 0.025
    }
    
    bayesian_result = requests.post('http://localhost:5000/predict', json={'features': sample_features, 'model': 'bayesian'}).json()
    cgnn_result = requests.post('http://localhost:5000/predict', json={'features': sample_features, 'model': 'cgnn'}).json()
    
    return jsonify({
        'sample_features': sample_features,
        'bayesian': bayesian_result,
        'cgnn': cgnn_result
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

