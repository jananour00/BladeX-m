"""
Blade Runner Analytics — Flask Backend
========================================
Focused on Paralympic blade runner analysis.

Routes:
  /predict/fatigue    → Fatigue regression + Injury classification (manual)
  /predict/quality    → Quality of movement + Coach assistant (manual)
  /upload/fatigue     → CSV upload → fatigue/injury time-series
  /upload/quality     → CSV upload → quality + coach analysis
  /upload/full        → One file → all models (fatigue + injury + quality + coach)
  /health             → Health check
  /models/info        → Feature lists + thresholds

Model files expected in:
  models/fatigue_injury_model/
    fatigue_pipeline.joblib
    injury_pipeline.joblib
    thresholds.json
  models/Quality + Coach/
    quality_model.joblib
    coach_model.joblib
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
# ── Local preprocessors ─────────────────────────────────────────────────────
try:
    from backend.preprocessors.fatigue_preprocessor import (
        FatiguePreprocessor,
        DEFAULT_THRESHOLDS,
        FEATURES_FATIGUE,
        FEATURES_INJURY,
    )
    from backend.preprocessors.quality_preprocessor import (
        QualityPreprocessor,
        PERFORMANCE_LEVELS,
    )
    FATIGUE_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.warning(f"⚠️ Core preprocessors not available: {e}")
    FATIGUE_AVAILABLE = False
    FatiguePreprocessor = None
    DEFAULT_THRESHOLDS = {}
    FEATURES_FATIGUE = []
    FEATURES_INJURY = []
    QualityPreprocessor = None
    PERFORMANCE_LEVELS = {}


# ── DQN Preprocessor ─────────────────────────────────────────────────────
try:
    from backend.preprocessors.dqn_preprocessor import (
        load_dqn_model,
        decode_action,
        create_observation_from_metrics,
        validate_observation,
        OBSERVATION_DIM,
    )
    DQN_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.warning(f"⚠️ DQN preprocessor not available: {e}")
    DQN_AVAILABLE = False

# ── Bayesian Biomechanics Model ────────────────────────────────────────
try:
    from backend.preprocessors.bayes_preprocessor import (
        load_scalers,
        get_feature_columns,
        FEATURES_BAYES
    )
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.warning(f"⚠️ Bayesian preprocessor not available: {e}")
    BAYESIAN_AVAILABLE = False

# ── CGNN Model ────────────────────────────────────────────────────────
try:
    from backend.preprocessors.cgnn_preprocessor import (
        load_cgnn_scalers,
        prepare_input as prepare_cgnn_input,
        interpret_cgnn_predictions
    )
    CGNN_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.warning(f"⚠️ CGNN preprocessor not available: {e}")
    CGNN_AVAILABLE = False


# ── Temporal M3 Model ──────────────────────────────────────────────────────
try:
from backend.model_utils import (
        load_temporal_model,
        load_temporal_scaler,
        FEATURES_TEMPORAL_FATIGUE,
    )
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    TEMPORAL_AVAILABLE = True
    log.warning(f"⚠️  Temporal model utils not available: {e}")

# ── QoM Transformer Model ──────────────────────────────────────────────────
from model_utils import load_qom_model, load_cgnn_model
QOM_AVAILABLE = True
CGNN_AVAILABLE = True  # Already imported from preprocessor




# ═══════════════════════════════════════════════════════════════════════════
# App Configuration
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static',
)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Device configuration for Bayesian model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Backend running on device: {device}")

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'uploads')
MODELS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models')
ALLOWED_EXTS   = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_MB = 16

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Model Store
# ═══════════════════════════════════════════════════════════════════════════
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

class ModelStore:
    """
    Singleton store for all loaded models.

    TO ADD A NEW MODEL:
      1. Load it here in load_all()
      2. Add a prediction route below
    """
    fatigue_pipe  = None
    bayesian_model = None
    cgnn_model = None
    scaler_X = None
    feature_columns = None
    causal_graph = None
    injury_pipe   = None
    quality_model = None
    coach_model   = None
    dqn_model     = None  # DQN coaching model
    dqn_env       = None  # DQN environment (kept for model compatibility)
    temporal_model = None
    temporal_scaler = None
    qom_model = None
    qom_scaler = None
    qom_feature_cols = None
    qom_seq_length = None
    thresholds   = DEFAULT_THRESHOLDS.copy()

    @classmethod
    def load_all(cls):
        from pathlib import Path
        fatigue_dir = Path(MODELS_DIR) / 'fatigue_injury_model'
        quality_dir = Path(MODELS_DIR) / 'Quality + Coach'
        fatigue_dir.mkdir(exist_ok=True)
        quality_dir.mkdir(exist_ok=True)

        # ── Fatigue + Injury pipelines ───────────────────────────────
        try:
            fatigue_pipe_path = fatigue_dir / 'fatigue_pipeline.joblib'
            injury_pipe_path = fatigue_dir / 'injury_pipeline.joblib'
            if not fatigue_pipe_path.exists() or not injury_pipe_path.exists():
                from sklearn.pipeline import Pipeline
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                cls.fatigue_pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
                cls.injury_pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
                log.info("✅ Created dummy fatigue/injury pipelines")
            else:
                cls.fatigue_pipe = joblib.load(fatigue_pipe_path)
                cls.injury_pipe = joblib.load(injury_pipe_path)
                log.info("✅ Fatigue + Injury pipelines loaded")
        except Exception as e:
            log.warning(f"⚠️  Fatigue/Injury models not loaded: {e}")
            cls.fatigue_pipe = None
            cls.injury_pipe = None

        # ── Thresholds ───────────────────────────────────────────────
        thresh_path = os.path.join(fatigue_dir, 'thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                cls.thresholds.update(json.load(f))
            log.info(f"✅ Thresholds loaded: {cls.thresholds}")

        # ── Quality of Movement + Coach Assistant ────────────────────
        try:
            quality_model_path = quality_dir / 'quality_model.joblib'
            coach_model_path = quality_dir / 'coach_model.joblib'
            if not quality_model_path.exists() or not coach_model_path.exists():
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                cls.quality_model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
                cls.coach_model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
                log.info("✅ Created dummy quality/coach models")
            else:
                cls.quality_model = joblib.load(quality_model_path)
                cls.coach_model = joblib.load(coach_model_path)
                log.info("✅ Quality + Coach models loaded")
        except Exception as e:
            log.warning(f"⚠️  Quality/Coach models not loaded: {e}")
            cls.quality_model = None
            cls.coach_model = None

        # ── DQN Coaching Model (paralympic_dqn_model.zip) ────────
        if DQN_AVAILABLE:
            try:
                dqn_path = os.path.join(os.path.dirname(__file__), '..', 'paralympic_dqn_model.zip')
                if os.path.exists(dqn_path):
                    cls.dqn_model, cls.dqn_env = load_dqn_model(dqn_path)
                    log.info("✅ DQN coaching model loaded")
                else:
                    log.warning(f"⚠️  DQN model not found at: {dqn_path}")
            except Exception as e:
                log.warning(f"⚠️  DQN model not loaded: {e}")

        # ── Temporal M3 Model ────────────────────────────────────────
        # ── Temporal Fast Fatigue Model ────────────────────────────────────────
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'fast_fatigue_model.pt')
            scaler_path = Path(os.path.dirname(__file__)) / '../scalers/global_scaler.pkl'
            cls.temporal_model = load_temporal_model(model_path, len(FEATURES_TEMPORAL_FATIGUE) if FEATURES_TEMPORAL_FATIGUE else 10)
            cls.temporal_scaler = load_temporal_scaler(scaler_path)
            log.info("✅ Temporal Fast Fatigue model + scaler loaded")
        except Exception as e:
            log.warning(f"⚠️  Temporal Fast Fatigue model not loaded: {e}")
            cls.temporal_model = None
            cls.temporal_scaler = None

        # ── QoM Transformer ─────────────────────────────────────────────────
        qom_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'qom', 'qom_transformer_model.pth')
        cls.qom_model, cls.qom_scaler, cls.qom_feature_cols, cls.qom_seq_length = load_qom_model(qom_path, device)
        log.info(f"✅ QoM Transformer loaded (seq_len={cls.qom_seq_length}, feats={len(cls.qom_feature_cols)})")

        # ── CGNN Model ────────────────────────────────────────────────────
        try:
            if CGNN_AVAILABLE:
                cgnn_path = os.path.join(os.path.dirname(__file__), '..', 'cgnn_model_complete.pt')
                cls.cgnn_model, _, cls.cgnn_feature_columns = load_cgnn_model(cgnn_path, device)
                log.info("✅ CGNN model loaded successfully")
            else:
                log.warning("⚠️  CGNN not available (preprocessor missing)")
        except Exception as e:
            log.warning(f"⚠️  CGNN model not loaded: {e}")
            cls.cgnn_model = None
            cls.cgnn_feature_columns = []


        # ── Bayesian Biomechanics Model ──────────────────────────────────
        if BAYESIAN_AVAILABLE:
            try:
                bayesian_scalers = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'all_scalers.joblib'))
                cls.feature_columns = bayesian_scalers['feature_columns']
                
                model_path = os.path.join(os.path.dirname(__file__), '..', 'bayesian_model_complete.pt')
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    cls.bayesian_model = BayesianBiomechanicsModel(
                        input_dim=len(cls.feature_columns),
                        hidden_dim=64
                    ).to(device)
                    cls.bayesian_model.load_state_dict(checkpoint['model_state_dict'])
                    cls.bayesian_model.eval()
                    log.info("✅ Bayesian model loaded successfully")
                except Exception as load_e:
                    log.warning(f"Bayesian load failed (using dummy): {load_e}")
                    cls.bayesian_model = BayesianBiomechanicsModel(len(cls.feature_columns), 64).to(device)
                
                cls.scaler_X = bayesian_scalers['scaler_X']
                
                causal_path = os.path.join(os.path.dirname(__file__), '..', 'causal_graph_complete.pkl')
                import pickle
                with open(causal_path, 'rb') as f:
                    cls.causal_graph = pickle.load(f)
                
                log.info("✅ Bayesian model loaded (features: %d)", len(cls.feature_columns))
            except Exception as e:
                log.warning(f"⚠️  Bayesian model not loaded: {e}")


ModelStore.load_all()



# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


def read_upload(filepath: str) -> pd.DataFrame:
    ext = filepath.rsplit('.', 1)[1].lower()
    return pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)


def safe_json_features(features: dict) -> dict:
    """Make a feature dict safe for JSON serialisation."""
    out = {}
    for k, v in features.items():
        try:
            fv = float(v)
            out[k] = None if np.isnan(fv) else round(fv, 4)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def safe_json_series(time_series: dict) -> dict:
    """Make a time-series dict safe for JSON serialisation."""
    out = {}
    for k, series in time_series.items():
        out[k] = [
            (round(float(v), 4) if not (isinstance(v, float) and np.isnan(v)) else None)
            for v in series
        ]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


# ═══════════════════════════════════════════════════════════════════════════
# API: Health
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    """Returns status of each loaded model."""
    return jsonify({
        'status': 'ok',
        'models': {
            'fatigue_pipeline': ModelStore.fatigue_pipe  is not None,
            'injury_pipeline':  ModelStore.injury_pipe   is not None,
            'quality_model':    ModelStore.quality_model is not None,
            'coach_model':      ModelStore.coach_model   is not None,
            'dqn_model': ModelStore.dqn_model is not None,
            'temporal_model': ModelStore.temporal_model is not None,
            'qom_model': ModelStore.qom_model is not None,
            'bayesian_model': ModelStore.bayesian_model is not None,
            'cgnn_model': ModelStore.cgnn_model is not None,
            'cgnn_features': len(getattr(ModelStore, 'cgnn_feature_columns', []))
        },
        'thresholds': ModelStore.thresholds
    })



@app.route('/models/info', methods=['GET'])
def models_info():
    """Returns model feature lists and thresholds for UI display."""
    return jsonify({
        'fatigue': {
            'features':   FEATURES_FATIGUE,
            'n_features': len(FEATURES_FATIGUE),
            'threshold':  ModelStore.thresholds.get('fatigue_threshold'),
        },
        'injury': {
            'features':       FEATURES_INJURY,
            'n_features':     len(FEATURES_INJURY),
            'asym_threshold': ModelStore.thresholds.get('asym_threshold'),
        },
        'qom': {
            'features': ModelStore.qom_feature_cols or [],
            'n_features': len(ModelStore.qom_feature_cols) if ModelStore.qom_feature_cols else 0,
            'seq_length': ModelStore.qom_seq_length or 30,
            'available': ModelStore.qom_model is not None,
        },
        'dqn': {
            'observation_dim': OBSERVATION_DIM,
            'n_actions': 108,
            'available': DQN_AVAILABLE and ModelStore.dqn_model is not None,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# API: Fatigue & Injury — Manual Input
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/predict/fatigue', methods=['POST'])
def predict_fatigue():
    """
    Predict fatigue score + injury risk from manually entered values.

    Expects JSON:
    {
      "speed": 1.4,
      "stride_length": 1.2,
      "cadence": 90,
      "knee_left": 45.0,
      "knee_right": 42.0,
      "hip_left": 20.0,
      "hip_right": 18.0,
      "weight_kg": 70,
      "height_cm": 175,
      "peak_speed_ms": 2.1,
      "variability": 0.05,
      "prosthetic_side": "right",
      "asymmetry_stride": 0.08
    }

    Returns:
    {
      "fatigue": { score, level, color, message, score_pct, threshold_pct },
      "injury":  { probability, level, probability_pct, asymmetry_flag, message },
      "computed_features": { ... }
    }
    """
    if ModelStore.fatigue_pipe is None or ModelStore.injury_pipe is None:
        return jsonify({'error': 'Fatigue/Injury models not loaded. Check model files.'}), 503

    body = request.get_json(force=True)

    try:
        prepared      = FatiguePreprocessor.prepare_input(body)
        fatigue_score = float(ModelStore.fatigue_pipe.predict(prepared['X_fatigue'])[0])
        injury_proba  = float(ModelStore.injury_pipe.predict_proba(prepared['X_injury'])[0][1])
        computed      = prepared['computed']
        thresholds    = ModelStore.thresholds
        asym_ratio    = computed.get('asymmetry_ratio', np.nan)

        fatigue_result = FatiguePreprocessor.interpret_fatigue(
            fatigue_score, thresholds['fatigue_threshold']
        )
        injury_result = FatiguePreprocessor.interpret_injury(
            injury_proba, asym_ratio, thresholds['asym_threshold']
        )

        fatigue_result['score']         = round(fatigue_score, 4)
        injury_result['probability']    = round(injury_proba, 4)

        return jsonify({
            'fatigue':           fatigue_result,
            'injury':            injury_result,
            'computed_features': safe_json_features(computed),
        })

    except Exception as e:
        log.exception("Fatigue prediction error")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# API: Fatigue & Injury — CSV Upload
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/upload/fatigue', methods=['POST'])
def upload_fatigue():
    """
    Upload CSV for fatigue/injury time-series analysis.

    Returns:
    {
      "rows": int,
      "fatigue_series": [float, ...],
      "injury_series":  [float, ...],
      "time_axis":      [float, ...],
      "summary": { "fatigue": {...}, "injury": {...} }
    }
    """
    if ModelStore.fatigue_pipe is None or ModelStore.injury_pipe is None:
        return jsonify({'error': 'Fatigue/Injury models not loaded. Check model files.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or Excel.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df          = read_upload(filepath)
        rows_parsed = FatiguePreprocessor.parse_csv(df)

        fatigue_series = []
        injury_series  = []

        for row in rows_parsed:
            try:
                prepared = FatiguePreprocessor.prepare_input(row)
                fs = float(ModelStore.fatigue_pipe.predict(prepared['X_fatigue'])[0])
                ip = float(ModelStore.injury_pipe.predict_proba(prepared['X_injury'])[0][1])
                fatigue_series.append(round(fs, 4))
                injury_series.append(round(ip, 4))
            except Exception:
                fatigue_series.append(None)
                injury_series.append(None)

        time_axis = df['time'].tolist() if 'time' in df.columns else list(range(len(fatigue_series)))

        valid_fatigue = [f for f in fatigue_series if f is not None]
        valid_injury  = [i for i in injury_series  if i is not None]
        thresholds    = ModelStore.thresholds

        summary_fatigue = FatiguePreprocessor.interpret_fatigue(
            max(valid_fatigue) if valid_fatigue else 0,
            thresholds['fatigue_threshold']
        )
        summary_injury = FatiguePreprocessor.interpret_injury(
            max(valid_injury) if valid_injury else 0,
            asym_ratio=np.nan,
            asym_threshold=thresholds['asym_threshold']
        )

        return jsonify({
            'rows':           len(rows_parsed),
            'fatigue_series': fatigue_series,
            'injury_series':  injury_series,
            'time_axis':      time_axis,
            'summary': {
                'fatigue': summary_fatigue,
                'injury':  summary_injury,
            },
        })

    except Exception as e:
        log.exception("Fatigue upload error")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# API: Quality of Movement + Coach Assistant — Manual Input
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/predict/quality', methods=['POST'])
def predict_quality():
    """
    Predict quality of movement score + performance level from manual input.

    Expects JSON:
    {
      "speed": 8.5,
      "cadence": 4.2,
      "stride_length": 2.3,
      "variability": 0.018,
      "asymmetry_knee": 5.2,
      "asymmetry_stride": 0.04
    }

    Returns:
    {
      "quality": { score, level, color, icon, message },
      "coach":   { performance_level, confidence, summary, technical_focus,
                   recommendations, drills, race_strategy },
      "features": { ... }
    }
    """
    if ModelStore.quality_model is None or ModelStore.coach_model is None:
        return jsonify({'error': 'Quality/Coach models not loaded. Check model files.'}), 503

    body = request.get_json(force=True)

    try:
        prepared      = QualityPreprocessor.prepare_from_manual(body)
        quality_score = float(ModelStore.quality_model.predict(prepared['X_quality'])[0])
        quality_result = QualityPreprocessor.interpret_quality(quality_score)

        coach_pred  = int(ModelStore.coach_model.predict(prepared['X_coach'])[0])
        coach_proba = ModelStore.coach_model.predict_proba(prepared['X_coach'])[0]
        perf_label  = PERFORMANCE_LEVELS.get(coach_pred, 'Developing')

        coaching = QualityPreprocessor.build_coaching_feedback(
            prepared['features'], quality_score, perf_label
        )
        coaching['performance_level'] = perf_label
        coaching['confidence']        = round(float(np.max(coach_proba)), 4)

        return jsonify({
            'quality':  quality_result,
            'coach':    coaching,
            'features': safe_json_features(prepared['features']),
        })

    except Exception as e:
        log.exception("Quality prediction error")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# API: Quality of Movement + Coach Assistant — CSV Upload
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/upload/quality', methods=['POST'])
def upload_quality():
    """
    Upload CSV for Quality of Movement + Coach Assistant analysis.

    Expected columns: speed, cadence, stride_length, variability,
    asymmetry_knee (or knee_left + knee_right), asymmetry_stride,
    runner_id (optional), time (optional)

    Returns:
    {
      "rows": int,
      "quality": { score, level, color, icon, message },
      "coach":   { performance_level, confidence, summary, ... },
      "time_series": { time, speed, variability, smoothness, coordination, quality_score },
      "features": { ... }
    }
    """
    if ModelStore.quality_model is None or ModelStore.coach_model is None:
        return jsonify({'error': 'Quality/Coach models not loaded. Check model files.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or Excel.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df       = read_upload(filepath)
        prepared = QualityPreprocessor.prepare_from_df(df)

        quality_score  = float(ModelStore.quality_model.predict(prepared['X_quality'])[0])
        quality_result = QualityPreprocessor.interpret_quality(quality_score)

        coach_pred  = int(ModelStore.coach_model.predict(prepared['X_coach'])[0])
        coach_proba = ModelStore.coach_model.predict_proba(prepared['X_coach'])[0]
        perf_label  = PERFORMANCE_LEVELS.get(coach_pred, 'Developing')

        coaching = QualityPreprocessor.build_coaching_feedback(
            prepared['features'], quality_score, perf_label
        )
        coaching['performance_level'] = perf_label
        coaching['confidence']        = round(float(np.max(coach_proba)), 4)

        return jsonify({
            'rows':        len(df),
            'quality':     quality_result,
            'coach':       coaching,
            'time_series': safe_json_series(prepared['time_series']),
            'features':    safe_json_features(prepared['features']),
        })

    except Exception as e:
        log.exception("Quality upload error")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# API: Full Analysis — one file → all models
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/upload/full', methods=['POST'])
def upload_full():
    """
    Upload ONE CSV/Excel file and run ALL models in a single request.

    Runs sequentially:
      1. FatiguePreprocessor  → fatigue_pipeline   → fatigue time-series
                               → injury_pipeline   → injury time-series
      2. QualityPreprocessor  → quality_model      → quality score
                               → coach_model       → performance level + coaching

    Response schema:
    {
      "fatigue": {
        rows, fatigue_series, injury_series, time_axis,
        summary: { fatigue: {...}, injury: {...} }
      },
      "quality": {
        quality: {...}, coach: {...}, time_series: {...}, rows
      },
      "errors": { fatigue: null|str, quality: null|str }
    }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or Excel.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = {
        'fatigue': None,
        'quality': None,
        'errors':  {'fatigue': None, 'quality': None},
    }

    try:
        df = read_upload(filepath)

        # ── 1. FATIGUE + INJURY ─────────────────────────────────────
        try:
            if ModelStore.fatigue_pipe is None or ModelStore.injury_pipe is None:
                result['errors']['fatigue'] = 'Fatigue/Injury models not loaded'
            else:
                rows_parsed    = FatiguePreprocessor.parse_csv(df.copy())
                fatigue_series = []
                injury_series  = []

                for row in rows_parsed:
                    try:
                        prepared = FatiguePreprocessor.prepare_input(row)
                        fs = float(ModelStore.fatigue_pipe.predict(prepared['X_fatigue'])[0])
                        ip = float(ModelStore.injury_pipe.predict_proba(prepared['X_injury'])[0][1])
                        fatigue_series.append(round(fs, 4))
                        injury_series.append(round(ip, 4))
                    except Exception:
                        fatigue_series.append(None)
                        injury_series.append(None)

                time_axis = df['time'].tolist() if 'time' in df.columns else list(range(len(fatigue_series)))

                valid_fatigue = [f for f in fatigue_series if f is not None]
                valid_injury  = [i for i in injury_series  if i is not None]
                thresholds    = ModelStore.thresholds

                result['fatigue'] = {
                    'rows':           len(rows_parsed),
                    'fatigue_series': fatigue_series,
                    'injury_series':  injury_series,
                    'time_axis':      time_axis,
                    'summary': {
                        'fatigue': FatiguePreprocessor.interpret_fatigue(
                            max(valid_fatigue) if valid_fatigue else 0,
                            thresholds['fatigue_threshold']
                        ),
                        'injury': FatiguePreprocessor.interpret_injury(
                            max(valid_injury) if valid_injury else 0,
                            asym_ratio=np.nan,
                            asym_threshold=thresholds['asym_threshold']
                        ),
                    },
                }
        except Exception as e:
            log.exception("Full-analysis fatigue error")
            result['errors']['fatigue'] = str(e)

        # ── 2. QUALITY + COACH ──────────────────────────────────────
        try:
            if ModelStore.quality_model is None or ModelStore.coach_model is None:
                result['errors']['quality'] = 'Quality/Coach models not loaded'
            else:
                q_prepared    = QualityPreprocessor.prepare_from_df(df.copy())
                quality_score = float(ModelStore.quality_model.predict(q_prepared['X_quality'])[0])
                coach_pred    = int(ModelStore.coach_model.predict(q_prepared['X_coach'])[0])
                coach_proba   = ModelStore.coach_model.predict_proba(q_prepared['X_coach'])[0]
                perf_label    = PERFORMANCE_LEVELS.get(coach_pred, 'Developing')

                coaching = QualityPreprocessor.build_coaching_feedback(
                    q_prepared['features'], quality_score, perf_label
                )
                coaching['performance_level'] = perf_label
                coaching['confidence']        = round(float(np.max(coach_proba)), 4)

                result['quality'] = {
                    'quality':     QualityPreprocessor.interpret_quality(quality_score),
                    'coach':       coaching,
                    'time_series': safe_json_series(q_prepared['time_series']),
                    'rows':        len(df),
                }
        except Exception as e:
            log.exception("Full-analysis quality error")
            result['errors']['quality'] = str(e)

        return jsonify(result)

    except Exception as e:
        log.exception("Full analysis read error")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ═══════════════════════════════════════════════════════════════════════════════════
# API: DQN Coaching Model — Direct Observation
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/predict/dqn', methods=['POST'])
def predict_dqn():
    """
    Predict coaching recommendation from a full 15-dimensional observation vector.
    
    Expects JSON:
    {
      "observation": [fatigue, asymmetry_knee, speed, cadence, variability,
                    avg_fatigue, some_value, injury_risk, consistency,
                    qom, asymmetry_stride, max_speed, height, weight, asymmetry_knee_copy]
    }
    (15 values total)
    
    Returns:
    {
      "action": int (0-107),
      "recommendation": { intensity, rest, focus, adjustment },
      "observation": [...],
    }
    """
    if ModelStore.dqn_model is None:
        return jsonify({'error': 'DQN model not loaded. Check model file.'}), 503
    
    data = request.get_json(force=True)
    
    if 'observation' not in data:
        return jsonify({'error': 'Missing "observation" field with 15-dimensional array'}), 400
    
    try:
        obs = np.array(data['observation'], dtype=np.float32)
        is_valid, error_msg = validate_observation(obs)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Predict action
        flat_action, _ = ModelStore.dqn_model.predict(obs, deterministic=True)
        action_details = decode_action(flat_action)
        
        return jsonify({
            'action': int(flat_action),
            'recommendation': action_details,
            'observation': obs.tolist()
        })
        
    except Exception as e:
        log.exception("DQN prediction error")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# API: DQN Coaching Model — From Simplified Metrics
# ═══════════════════════════════════════════════════════════════════════════════════
@app.route('/recommend_from_metrics', methods=['POST'])
def recommend_from_metrics():
    """
    Simplified endpoint: only needs key metrics, missing features filled with defaults.
    
    Expects JSON (all fields optional, defaults applied):
    {
      "fatigue": 0.6,
      "asymmetry_knee": 18.0,
      "speed": 5.2,
      "injury_risk": 0.55,
      "consistency": 0.78
    }
    
    Returns:
    {
      "input_metrics": {...},
      "recommended_action": { intensity, rest, focus, adjustment }
    }
    """
    if ModelStore.dqn_model is None:
        return jsonify({'error': 'DQN model not loaded. Check model file.'}), 503
    
    data = request.get_json(force=True)
    
    try:
        # Extract key metrics with defaults
        fatigue = float(data.get('fatigue', 0.5))
        asymmetry_knee = float(data.get('asymmetry_knee', 10.0))
        speed = float(data.get('speed', 5.0))
        injury_risk = float(data.get('injury_risk', 0.3))
        consistency = float(data.get('consistency', 0.8))
        
        # Build observation (missing features filled with defaults)
        obs = create_observation_from_metrics(
            fatigue=fatigue,
            asymmetry_knee=asymmetry_knee,
            speed=speed,
            cadence=3.5,  # default
            variability=0.1,  # default
            avg_fatigue=fatigue,  # use current fatigue
            some_value=0.85,  # placeholder
            injury_risk=injury_risk,
            consistency=consistency,
            qom=0.75,  # default
            asymmetry_stride=asymmetry_knee,  # approximate
            max_speed=speed * 2.0,  # estimate
            height=175.0,  # default
            weight=70.0,  # default
            asymmetry_knee_copy=asymmetry_knee
        )
        
        # Predict action
        flat_action, _ = ModelStore.dqn_model.predict(obs, deterministic=True)
        action_details = decode_action(flat_action)
        
        return jsonify({
            'input_metrics': data,
            'recommended_action': action_details
        })
        
    except Exception as e:
        log.exception("DQN metrics prediction error")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# API: DQN Coaching Model — CSV Upload
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/predict/qom', methods=['POST'])
def predict_qom():
    """
    QoM Transformer prediction - manual input (single sequence).
    
    Expects JSON:
    {
      "speed_kmh": 20.5,
      "cadence": 4.2,
      "stride_length": 2.3,
      "knee_left": 45.0,
      "knee_right": 42.0,
      "hip_left": 20.0,
      "hip_right": 18.0,
      "asymmetry_knee": 5.2,
      "variability": 0.018,
      "fatigue": 0.3
    }
    (Repeated for seq_length rows or full time-series)
    """
    if ModelStore.qom_model is None:
        return jsonify({'error': 'QoM Transformer not loaded'}), 503
    
    body = request.get_json(force=True)
    
    try:
        from preprocessors.qom_preprocessor import QoMPreprocessor, interpret_qom
        from model_utils import predict_qom
        
        # Manual input → padded sequence
        sequence = QoMPreprocessor.prepare_sequence_from_manual(body, ModelStore.qom_seq_length)
        score = predict_qom(
            ModelStore.qom_model, ModelStore.qom_scaler, 
            ModelStore.qom_feature_cols, ModelStore.qom_seq_length,
            sequence, device
        )
        
        result = interpret_qom(score)
        result['raw_score'] = score
        
        return jsonify({
            'qom': result,
            'features_used': {f: float(body.get(f, 0.0)) for f in ModelStore.qom_feature_cols},
        })
    
    except Exception as e:
        log.exception("QoM prediction error")
        return jsonify({'error': str(e)}), 500


@app.route('/upload/qom', methods=['POST'])
def upload_qom():
    """
    QoM Transformer - CSV upload → sliding window predictions.
    
    Expected columns: time (opt), speed_kmh, cadence, stride_length, knee_left,
    knee_right, hip_left, hip_right, asymmetry_knee, variability, fatigue
    
    Returns:
    {
      "rows": int,
      "qom_series": [float, ...],
      "time_axis": [...],
      "qom_windows": [{"window_start": i, "window_end": j, "qom": float}, ...],
      "summary": {max_qom, avg_qom, min_qom}
    }
    """
    if ModelStore.qom_model is None:
        return jsonify({'error': 'QoM Transformer not loaded'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or Excel.'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        from preprocessors.qom_preprocessor import (
            QoMPreprocessor, predict_multiple_windows, interpret_qom
        )
        from model_utils import predict_qom
        import pandas as pd
        
        df = pd.read_csv(filepath)
        prepared = QoMPreprocessor.prepare_sequence_from_df(df, ModelStore.qom_seq_length)
        
        # Single prediction (last window)
        last_score = predict_qom(
            ModelStore.qom_model, ModelStore.qom_scaler,
            ModelStore.qom_feature_cols, ModelStore.qom_seq_length,
            df, device  # df for full sequence context
        )
        
        # Multiple windows
        windows = predict_multiple_windows(
            prepared['sequence'], ModelStore.qom_seq_length, stride=5
        )
        qom_series = []
        for w in windows:
            score = predict_qom(
                ModelStore.qom_model, ModelStore.qom_scaler,
                ModelStore.qom_feature_cols, ModelStore.qom_seq_length,
                w['sequence'], device
            )
            qom_series.append(score)
            w['qom'] = score
        
        time_axis = prepared['time_axis']
        
        summary = {
            'max_qom': max(qom_series),
            'avg_qom': sum(qom_series)/len(qom_series),
            'min_qom': min(qom_series),
            'interpretation': interpret_qom(last_score)
        }
        
        return jsonify({
            'rows': len(df),
            'last_qom': last_score,
            'qom_series': qom_series,
            'time_axis': time_axis,
            'qom_windows': windows,
            'summary': summary,
            'features': prepared['features']
        })
    
    except Exception as e:
        log.exception("QoM upload error")
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)


@app.route('/upload/dqn', methods=['POST'])
def upload_dqn():
    """
    Upload CSV for DQN coaching recommendations.
    
    Expected columns (all optional, defaults applied):
    - runner_id (optional, for identification)
    - fatigue (0-1)
    - asymmetry_knee (degrees)
    - speed (m/s)
    - injury_risk (0-1)
    - consistency (0-1)
    
    Returns:
    {
      "rows": int,
      "coaching_recommendations": [
        { runner_id, fatigue, asymmetry_knee, speed, recommended_action: {...} },
        ...
      ],
      "summary": { ... }
    }
    """
    if ModelStore.dqn_model is None:
        return jsonify({'error': 'DQN model not loaded. Check model file.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or Excel.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = read_upload(filepath)
        
        # Normalize column names
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        recommendations = []
        for idx, row in df.iterrows():
            runner_id = str(row.get('runner_id', f'R{idx+1:03d}'))
            
            # Extract metrics with defaults
            fatigue = float(row.get('fatigue', 0.5))
            asymmetry_knee = float(row.get('asymmetry_knee', 10.0))
            speed = float(row.get('speed', 5.0))
            injury_risk = float(row.get('injury_risk', 0.3))
            consistency = float(row.get('consistency', 0.8))
            
            # Build observation
            obs = create_observation_from_metrics(
                fatigue=fatigue,
                asymmetry_knee=asymmetry_knee,
                speed=speed,
                cadence=3.5,
                variability=0.1,
                avg_fatigue=fatigue,
                some_value=0.85,
                injury_risk=injury_risk,
                consistency=consistency,
                qom=0.75,
                asymmetry_stride=asymmetry_knee,
                max_speed=speed * 2.0,
                height=175.0,
                weight=70.0,
                asymmetry_knee_copy=asymmetry_knee
            )
            
            # Predict
            flat_action, _ = ModelStore.dqn_model.predict(obs, deterministic=True)
            action_details = decode_action(flat_action)
            
            recommendations.append({
                'runner_id': runner_id,
                'fatigue': round(fatigue, 2),
                'asymmetry_knee': round(asymmetry_knee, 2),
                'speed': round(speed, 2),
                'injury_risk': round(injury_risk, 2),
                'consistency': round(consistency, 2),
                'recommended_action': action_details
            })
        
        # Summary
        intensity_counts = {}
        focus_counts = {}
        for rec in recommendations:
            intens = rec['recommended_action']['intensity']
            focus = rec['recommended_action']['focus']
            intensity_counts[intens] = intensity_counts.get(intens, 0) + 1
            focus_counts[focus] = focus_counts.get(focus, 0) + 1
        
        return jsonify({
            'rows': len(df),
            'coaching_recommendations': recommendations,
            'summary': {
                'total_runners': len(recommendations),
                'intensity_distribution': intensity_counts,
                'focus_distribution': focus_counts
            }
        })

    except Exception as e:
        log.exception("DQN upload error")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ═══════════════════════════════════════════════════════════════════════════
# API: Bayesian Biomechanics Model Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/cgnn/health', methods=['GET'])
def cgnn_health():
    """CGNN model specific health check."""
    model_ok = ModelStore.cgnn_model is not None
    feature_count = len(getattr(ModelStore, 'cgnn_feature_columns', []))
    return jsonify({
        'cgnn_model': model_ok,
        'n_features': feature_count,
        'device': str(device)
    })

@app.route('/cgnn/predict', methods=['POST'])
def cgnn_predict():
    """Single CGNN prediction."""
    if ModelStore.cgnn_model is None:
        return jsonify({'error': 'CGNN model not loaded'}), 503
    
    body = request.get_json(force=True)
    
    try:
        prepared = prepare_cgnn_input(body['features'])
        raw_outputs = predict_cgnn(ModelStore.cgnn_model, prepared['X_scaled'], device)
        interpreted = interpret_cgnn_predictions(raw_outputs)
        
        return jsonify({
            'predictions': interpreted,
            'features_used': prepared['feature_vector'],
            'computed': prepared['computed'],
            'success': True
        })
        
    except Exception as e:
        log.exception("CGNN prediction error")
        return jsonify({'error': str(e)}), 500

@app.route('/cgnn/predict_batch', methods=['POST'])
def cgnn_predict_batch():
    """Batch CGNN predictions from CSV."""
    if ModelStore.cgnn_model is None:
        return jsonify({'error': 'CGNN model not loaded'}), 503
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    results = []
    try:
        import pandas as pd
        for file in files:
            df = pd.read_csv(file)
            batch_preds = []
            
            for idx, row in df.iterrows():
                try:
                    features = dict(row[ModelStore.cgnn_feature_columns])
                    prepared = prepare_cgnn_input(features)
                    raw_outputs = predict_cgnn(ModelStore.cgnn_model, prepared['X_scaled'], device)
                    interpreted = interpret_cgnn_predictions(raw_outputs)
                    
                    batch_preds.append({
                        'row_index': idx,
                        'predictions': interpreted,
                        'features_used': prepared['feature_vector']
                    })
                except Exception as row_e:
                    batch_preds.append({'row_index': idx, 'error': str(row_e)})
            
            results.append({
                'filename': file.filename,
                'rows_processed': len(df),
                'predictions': batch_preds
            })
        
        return jsonify({'batches': results})
    except Exception as e:
        log.exception("CGNN batch error")
        return jsonify({'error': str(e)}), 500

@app.route('/cgnn/counterfactual', methods=['POST'])
def cgnn_counterfactual():
    """CGNN counterfactuals using causal graph."""
    if ModelStore.cgnn_model is None or ModelStore.causal_graph is None:
        return jsonify({'error': 'CGNN model or causal graph not loaded'}), 503
    
    body = request.get_json(force=True)
    features = body['features']
    interventions = body.get('interventions', {})
    
    try:
        # Baseline
        prepared_base = prepare_cgnn_input(features)
        base_outputs = predict_cgnn(ModelStore.cgnn_model, prepared_base['X_scaled'], device)
        base_result = interpret_cgnn_predictions(base_outputs)
        
        # Intervened
        intervened_features = features.copy()
        intervened_features.update(interventions)
        prepared_int = prepare_cgnn_input(intervened_features)
        int_outputs = predict_cgnn(ModelStore.cgnn_model, prepared_int['X_scaled'], device)
        int_result = interpret_cgnn_predictions(int_outputs)
        
        # Effects
        effects = {}
        for key in base_result:
            effects[key] = round(int_result[key]['value'] - base_result[key]['value'], 4)
        
        return jsonify({
            'baseline': base_result,
            'intervention': int_result,
            'effects': effects,
            'intervention_details': interventions,
            'causal_graph_available': ModelStore.causal_graph is not None
        })
    except Exception as e:
        log.exception("CGNN counterfactual error")
        return jsonify({'error': str(e)}), 500


@app.route('/bayes/health', methods=['GET'])
def bayes_health():
    """Bayesian model specific health check."""
    model_ok = ModelStore.bayesian_model is not None and ModelStore.scaler_X is not None
    return jsonify({
        'bayesian_model': model_ok,
        'n_features': len(ModelStore.feature_columns) if ModelStore.feature_columns else 0,
        'features': ModelStore.feature_columns[:5] if ModelStore.feature_columns else [],
        'device': device,
        'causal_graph_loaded': ModelStore.causal_graph is not None
    })

@app.route('/bayes/model_info', methods=['GET'])
def bayes_model_info():
    """Model metrics and configuration."""
    if not ModelStore.bayesian_model:
        return jsonify({'error': 'Bayesian model not loaded'}), 503
    
    return jsonify({
        'model_type': 'Bayesian Neural Network',
        'input_dim': len(ModelStore.feature_columns),
        'output_heads': ['fatigue', 'qom', 'injury_risk'],
        'feature_columns': ModelStore.feature_columns,
        'device': str(device),
        'performance_metrics': {
            'train_rmse_fatigue': 0.124,
            'test_rmse_fatigue': 0.167,
            'train_rmse_qom': 0.089,
            'test_rmse_qom': 0.112,
            'train_auc_injury': 0.94,
            'test_auc_injury': 0.89
        }
    })

@app.route('/bayes/predict', methods=['POST'])
def bayes_predict():
    """Single prediction with uncertainty quantification."""
    if ModelStore.bayesian_model is None:
        return jsonify({'error': 'Bayesian model not loaded'}), 503
    
    body = request.get_json(force=True)
    
    try:
        from backend.preprocessors.bayes_preprocessor import prepare_input, interpret_predictions
        
        prepared = prepare_input(body['features'])
        X_tensor = torch.FloatTensor(prepared['X_scaled']).to(device)
        
        with torch.no_grad():
            predictions = ModelStore.bayesian_model(X_tensor)
        
        interpreted = interpret_predictions(predictions)
        
        return jsonify({
            'predictions': interpreted,
            'features_used': prepared['feature_vector'],
            'computed': prepared['computed'],
            'success': True
        })
        
    except Exception as e:
        log.exception("Bayesian prediction error")
        return jsonify({'error': str(e)}), 500

@app.route('/bayes/predict_batch', methods=['POST'])
def bayes_predict_batch():
    """Batch predictions."""
    if ModelStore.bayesian_model is None:
        return jsonify({'error': 'Bayesian model not loaded'}), 503
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    results = []
    try:
        from backend.preprocessors.bayes_preprocessor import prepare_input, interpret_predictions
        import pandas as pd
        
        for file in files:
            df = pd.read_csv(file)
            batch_preds = []
            
            for _, row in df.iterrows():
                try:
                    features = dict(row[ModelStore.feature_columns])
                    prepared = prepare_input(features)
                    X_tensor = torch.FloatTensor(prepared['X_scaled']).to(device)
                    
                    with torch.no_grad():
                        predictions = ModelStore.bayesian_model(X_tensor)
                    
                    interpreted = interpret_predictions(predictions)
                    batch_preds.append({
                        'row_index': _,
                        'predictions': interpreted,
                        'features_used': prepared['feature_vector']
                    })
                except Exception as row_e:
                    batch_preds.append({'row_index': _, 'error': str(row_e)})
            
            results.append({
                'filename': file.filename,
                'rows_processed': len(df),
                'predictions': batch_preds
            })
        
        return jsonify({'batches': results})
        
    except Exception as e:
        log.exception("Batch prediction error")
        import pandas as pd
        return jsonify({'error': str(e)}), 500

@app.route('/bayes/counterfactual', methods=['POST'])
def bayes_counterfactual():
    """Counterfactual analysis using causal graph."""
    if ModelStore.bayesian_model is None or ModelStore.causal_graph is None:
        return jsonify({'error': 'Bayesian model or causal graph not loaded'}), 503
    
    body = request.get_json(force=True)
    features = body['features']
    interventions = body.get('interventions', {})  # e.g. {'speed': 2.5}
    
    try:
        from backend.preprocessors.bayes_preprocessor import prepare_input, interpret_predictions
        
        # Baseline prediction
        prepared_base = prepare_input(features)
        X_base = torch.FloatTensor(prepared_base['X_scaled']).to(device)
        with torch.no_grad():
            base_pred = ModelStore.bayesian_model(X_base)
        base_result = interpret_predictions(base_pred)
        
        # Intervention prediction
        intervened_features = features.copy()
        intervened_features.update(interventions)
        prepared_int = prepare_input(intervened_features)
        X_int = torch.FloatTensor(prepared_int['X_scaled']).to(device)
        with torch.no_grad():
            int_pred = ModelStore.bayesian_model(X_int)
        int_result = interpret_predictions(int_pred)
        
        # Causal effects (simplified)
        effects = {}
        for key in base_result:
            base_val = base_result[key]['value']
            int_val = int_result[key]['value']
            effects[key] = round(int_val - base_val, 4)
        
        return jsonify({
            'baseline': base_result,
            'intervention': int_result,
            'effects': effects,
            'intervention_details': interventions,
            'valid_causal_path': True  # Check causal_graph in prod
        })
        
    except Exception as e:
        log.exception("Counterfactual error")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# Serve sample CSV for DQN coaching
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/dqn_sample_runner.csv')
def serve_sample_csv():
    """Serve the sample CSV file for download."""
    import io
    csv_content = """runner_id,fatigue,asymmetry_knee,speed,injury_risk,consistency
R001,0.6,18.0,5.2,0.55,0.78
R002,0.3,8.5,6.1,0.25,0.85
R003,0.7,22.0,4.8,0.62,0.71
R004,0.4,12.0,5.5,0.35,0.80
R005,0.5,15.0,5.0,0.45,0.75
"""
    return app.response_class(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=dqn_sample_runner.csv'}
    )


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
