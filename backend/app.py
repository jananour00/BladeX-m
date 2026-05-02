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
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
# ── Local preprocessors ─────────────────────────────────────────────────────
from preprocessors.fatigue_preprocessor import (
    FatiguePreprocessor,
    DEFAULT_THRESHOLDS,
    FEATURES_FATIGUE,
    FEATURES_INJURY,
)
from preprocessors.quality_preprocessor import (
    QualityPreprocessor,
    PERFORMANCE_LEVELS,
    
)

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
class ModelStore:
    """
    Singleton store for all loaded models.

    TO ADD A NEW MODEL:
      1. Load it here in load_all()
      2. Add a prediction route below
    """
    fatigue_pipe  = None
    injury_pipe   = None
    quality_model = None
    coach_model   = None
    thresholds    = DEFAULT_THRESHOLDS.copy()

    @classmethod
    def load_all(cls):
        fatigue_dir = os.path.join(MODELS_DIR, 'fatigue_injury_model')
        quality_dir = os.path.join(MODELS_DIR, 'Quality + Coach')

        # ── Fatigue + Injury pipelines ───────────────────────────────
        try:
            cls.fatigue_pipe = joblib.load(os.path.join(fatigue_dir, 'fatigue_pipeline.joblib'))
            cls.injury_pipe  = joblib.load(os.path.join(fatigue_dir, 'injury_pipeline.joblib'))
            log.info("✅ Fatigue + Injury pipelines loaded")
        except Exception as e:
            log.warning(f"⚠️  Fatigue/Injury models not loaded: {e}")

        # ── Thresholds ───────────────────────────────────────────────
        thresh_path = os.path.join(fatigue_dir, 'thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                cls.thresholds.update(json.load(f))
            log.info(f"✅ Thresholds loaded: {cls.thresholds}")

        # ── Quality of Movement + Coach Assistant ────────────────────
        try:
            cls.quality_model = joblib.load(os.path.join(quality_dir, 'quality_model.joblib'))
            cls.coach_model   = joblib.load(os.path.join(quality_dir, 'coach_model.joblib'))
            log.info("✅ Quality + Coach models loaded")
        except Exception as e:
            log.warning(f"⚠️  Quality/Coach models not loaded: {e}")


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
        },
        'thresholds': ModelStore.thresholds,
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


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
