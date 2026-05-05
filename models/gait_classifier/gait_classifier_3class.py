"""
Gait Pattern Classifier — 3-Class (Healthy / Amputee K2 / Amputee K3)
PRODUCTION GRADE VERSION - WITH PROPER PREDICTION

CRITICAL FIXES:
✅ Robust feature extraction (no hardcoded indices)
✅ Dynamic feature engineering
✅ Advanced augmentation (time scaling, magnitude warping, window slicing)
✅ Proper nested CV with stacking
✅ Already has calibration (keep it)
✅ FIXED: Proper prediction from existing subject data
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
FILES = {
    'subjects':      "File 01 Subjects information.xlsx",
    'spatiotemporal':"File 02 Spatio-temporal parameters.xlsx",
    'angles':        "File 03 lower limb joint angles in sagittal plane.xlsx",
    'grf':           "File 04 ground reaction force compnents.xlsx",
    'moments':       "File 05 lower limb joint moments in sagittal plane.xlsx",
}

K_LEVEL_MAP = {
    'AMPF1': 'amputee_K3', 'AMPF2': 'amputee_K3', 'AMPF3': 'amputee_K3',
    'AMPF4': 'amputee_K3', 'AMPF5': 'amputee_K2',
    'AMPM1': 'amputee_K2', 'AMPM2': 'amputee_K2', 'AMPM3': 'amputee_K3',
    'AMPM4': 'amputee_K3', 'AMPM5': 'amputee_K3', 'AMPM6': 'amputee_K2',
    'AMPM7': 'amputee_K3', 'AMPM8': 'amputee_K2', 'AMPM9': 'amputee_K3',
}

ANGLE_PAIRS = [('N_AFE', 'AMP_AFE'), ('N_KFE', 'AMP_KFE'), ('N_HIPFE', 'AMP_HIPFE')]
GRF_PAIRS = [('N_GRF_Vertical ', 'AMP_GRF_Vertical'), ('N_GRF_AP', 'AMP_GRF_AP'), ('N_GRF_ML', 'AMP_GRF_ML')]
MOMENT_PAIRS = [('N_Moment_AFE', 'AMP_Moment_AFE'), ('N_Moment_KFE', 'AMP_Moment_KFE'), ('N_Moment_HIPFE', 'AMP_Moment_HIPFE')]
SKIP = {'nan', '', 'NaN', 'mean', 'SD', 'R = Rihgt side', 'L = Left Side', 'P =Prosthetic side', 'S = Sound Side'}

# Global variable to store subject_data for prediction
SUBJECT_DATA_CACHE = None

# ─────────────────────────────────────────────
# 1. ROBUST FEATURE EXTRACTION (NO hardcoded indices)
# ─────────────────────────────────────────────
def normalize_to_101_points(signal):
    """
    Normalize any signal to exactly 101 points.
    Uses interpolation - robust for different lengths.
    """
    s = np.array(signal, dtype=float)
    valid = ~np.isnan(s)
    
    if valid.sum() < 5:
        return np.full(101, np.nan)
    
    # Interpolate to fill NaNs first
    if valid.sum() < len(s):
        x = np.arange(len(s))
        s_interp = np.interp(x, x[valid], s[valid])
    else:
        s_interp = s.copy()
    
    # Resample to exactly 101 points
    original_indices = np.linspace(0, 1, len(s_interp))
    target_indices = np.linspace(0, 1, 101)
    
    f = interp1d(original_indices, s_interp, kind='cubic', fill_value='extrapolate')
    normalized = f(target_indices)
    
    return normalized


def extract_robust_features(signal_normalized):
    """
    Extract features from normalized signal (exactly 101 points).
    No hardcoded indices - uses percentile-based positions.
    """
    s = signal_normalized
    if np.any(np.isnan(s)):
        return [np.nan] * 20
    
    # 1. Basic statistics
    mean_val = np.mean(s)
    std_val = np.std(s)
    max_val = np.max(s)
    min_val = np.min(s)
    rom = max_val - min_val
    area = np.trapz(s)
    
    # 2. Percentile-based points (instead of hardcoded 25, 60, 75)
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(s, p) for p in percentiles]
    
    # 3. Dynamic time points (based on gait phases)
    # Find peaks and troughs
    peaks, _ = find_peaks(s, height=np.mean(s), distance=10)
    troughs, _ = find_peaks(-s, distance=10)
    
    if len(peaks) > 0:
        first_peak_idx = peaks[0] / len(s)
        peak_value = s[peaks[0]]
    else:
        first_peak_idx = 0.5
        peak_value = max_val
    
    if len(troughs) > 0:
        first_trough_idx = troughs[0] / len(s)
        trough_value = s[troughs[0]]
    else:
        first_trough_idx = 0.25
        trough_value = min_val
    
    # 4. Slope features (rate of change)
    slopes = np.diff(s)
    mean_slope = np.mean(slopes)
    max_slope = np.max(slopes)
    min_slope = np.min(slopes)
    slope_variance = np.var(slopes)
    
    # 5. Energy and entropy
    energy = np.sum(s ** 2) / len(s)
    hist, _ = np.histogram(s, bins=10)
    hist = hist[hist > 0] / len(s)
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    
    # 6. Gait symmetry (compare first 50% with last 50%)
    half = len(s) // 2
    symmetry = np.corrcoef(s[:half], s[half:half*2])[0, 1] if half > 0 else 0
    
    # 7. Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(s - np.mean(s))))
    zcr = zero_crossings / len(s)
    
    # Combine all features
    features = [
        mean_val, std_val, max_val, min_val, rom, area,
        first_peak_idx, peak_value, first_trough_idx, trough_value,
        mean_slope, max_slope, min_slope, slope_variance,
        energy, entropy, symmetry, zcr,
    ]
    
    # Add percentile values
    features.extend(percentile_values)
    
    # Total: 18 + 5 = 23 features per signal
    return features


def extract_all_features_from_subject(signals):
    """
    Extract features from all 9 signals for a subject.
    """
    all_features = []
    
    for signal in signals:
        normalized = normalize_to_101_points(signal)
        features = extract_robust_features(normalized)
        all_features.extend(features)
    
    return all_features


# ─────────────────────────────────────────────
# 2. ADVANCED DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment_signal_advanced(signal, methods=None):
    """
    Advanced signal augmentation with multiple methods.
    """
    if methods is None:
        methods = ['noise', 'time_warp', 'magnitude_warp', 'smooth', 'scale']
    
    s = np.array(signal, dtype=float)
    valid = ~np.isnan(s)
    
    if valid.sum() < len(s) * 0.5:
        return [s]
    
    # Fill NaNs for augmentation
    if np.any(np.isnan(s)):
        x = np.arange(len(s))
        s = np.interp(x, x[valid], s[valid])
    
    augmented = [s]
    
    for method in methods:
        if method == 'noise':
            # Gaussian noise with adaptive level
            noise_level = np.random.uniform(0.005, 0.02) * np.std(s)
            augmented.append(s + np.random.normal(0, noise_level, len(s)))
        
        elif method == 'time_warp':
            # Non-linear time warping
            if len(s) > 20:
                warp = np.cumsum(np.random.uniform(0.9, 1.1, len(s)))
                warp = warp / warp[-1] * len(s)
                indices = np.clip(np.floor(warp).astype(int), 0, len(s)-1)
                augmented.append(s[indices])
        
        elif method == 'magnitude_warp':
            # Random magnitude scaling
            scale = np.random.uniform(0.8, 1.2)
            augmented.append(s * scale)
        
        elif method == 'smooth':
            # Gaussian smoothing with random sigma
            sigma = np.random.uniform(0.5, 2.0)
            augmented.append(gaussian_filter1d(s, sigma=sigma))
        
        elif method == 'scale':
            # Time scaling (stretch/compress)
            scale_factor = np.random.uniform(0.9, 1.1)
            new_len = int(len(s) * scale_factor)
            if new_len > 10:
                indices = np.linspace(0, len(s)-1, new_len)
                scaled = np.interp(indices, np.arange(len(s)), s)
                # Resample back to original length
                augmented.append(np.interp(np.arange(len(s)), indices, scaled))
    
    return augmented


def load_subject_data():
    """
    Load all subject data from Excel files and return the subject_data dictionary.
    This function is used by both training and prediction.
    """
    print("=" * 60)
    print("Loading Subject Data from Excel Files")
    print("=" * 60)
    
    subject_data = {}
    
    # Check if files exist first
    for key, path in FILES.items():
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            return None
    
    # Extract from Excel files
    for file_key, pairs in [('angles', ANGLE_PAIRS), ('grf', GRF_PAIRS), ('moments', MOMENT_PAIRS)]:
        try:
            all_sheets = pd.read_excel(FILES[file_key], sheet_name=None)
        except Exception as e:
            print(f"  Warning: Could not read {FILES[file_key]}: {e}")
            continue
        
        # Clean sheet names (remove spaces)
        clean_sheets = {k.strip(): v for k, v in all_sheets.items()}

        for normal_sheet, amp_sheet in pairs:
            for sheet_key, default_label in [(normal_sheet, 'healthy'), (amp_sheet, 'amputee_placeholder')]:
        
                sheet_key_clean = sheet_key.strip()

                if sheet_key_clean not in clean_sheets:
                    print(f"  ❌ Sheet not found: {sheet_key_clean}")
                    continue

                df = clean_sheets[sheet_key_clean]
                hdr = df.iloc[0]
                data = df.iloc[1:].reset_index(drop=True)
                max_cols = min(data.shape[1], 50)
                
                for ci in range(1, max_cols):
                    raw_name = str(hdr.iloc[ci]).strip().replace("'", "").replace(" ", "")
                    if raw_name in SKIP or raw_name == '':
                        continue
                    
                    subj_key = raw_name.upper()
                    if subj_key.startswith('AMP'):
                        subj_key = subj_key.split('.')[0].split('_')[0]
                    
                    label = 'healthy' if default_label == 'healthy' else K_LEVEL_MAP.get(subj_key, 'amputee_K3')
                    
                    vals = pd.to_numeric(data.iloc[:, ci], errors='coerce').values.astype(float)
                    
                    if subj_key not in subject_data:
                        subject_data[subj_key] = {'label': label, 'signals': []}
                    
                    subject_data[subj_key]['signals'].append(vals)
    
    # Add spatiotemporal
    try:
        st_normal = pd.read_excel(FILES['spatiotemporal'], sheet_name="Normal ")
        st_amp = pd.read_excel(FILES['spatiotemporal'], sheet_name="AMP")
        
        for df in [st_normal, st_amp]:
            for _, row in df.iterrows():
                code = str(row.iloc[0]).strip()
                if code in SKIP or code == '':
                    continue
                
                step_width, cycle_dur, velocity = np.nan, np.nan, np.nan
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'step width' in col_str:
                        step_width = pd.to_numeric(row[col], errors='coerce')
                    elif 'cycle duration' in col_str or 'gait cycle' in col_str:
                        cycle_dur = pd.to_numeric(row[col], errors='coerce')
                    elif 'velocity' in col_str:
                        velocity = pd.to_numeric(row[col], errors='coerce')
                
                key = code.upper()
                if key in subject_data:
                    subject_data[key]['step_width'] = step_width
                    subject_data[key]['cycle_duration'] = cycle_dur
                    subject_data[key]['velocity'] = velocity
    except Exception as e:
        print(f"  Warning: Spatiotemporal issue: {e}")
    
    print(f"\n  ✅ Loaded {len(subject_data)} subjects")
    return subject_data


def build_feature_matrix_with_advanced_augmentation():
    """
    Build feature matrix with advanced augmentation.
    """
    global SUBJECT_DATA_CACHE
    
    print("=" * 60)
    print("Building Feature Matrix (Advanced Augmentation)")
    print("=" * 60)
    
    subject_data = load_subject_data()
    
    if subject_data is None:
        return None, None, None
    
    SUBJECT_DATA_CACHE = subject_data
    
    X_list = []
    y_list = []
    subject_groups = []
    
    for subj_idx, (subj_key, data) in enumerate(subject_data.items()):
        signals = data['signals']
        
        # Ensure we have enough signals
        if len(signals) == 0:
            continue
        
        # If less than 9, pad with NaN signals
        while len(signals) < 9:
            signals.append(np.full(50, np.nan))
        
        # Original features
        original_features = extract_all_features_from_subject(signals)
        original_features.append(data.get('step_width', np.nan))
        original_features.append(data.get('cycle_duration', np.nan))
        original_features.append(data.get('velocity', np.nan))
        
        X_list.append(original_features)
        y_list.append(data['label'])
        subject_groups.append(subj_idx)
        
        # Advanced augmentation for minority classes
        if data['label'] in ['amputee_K2', 'amputee_K3']:
            for aug_idx in range(1, 4):  # 3 augmented versions
                augmented_signals = []
                
                # Augment each signal with different methods
                methods_options = [
                    ['noise', 'time_warp'],
                    ['magnitude_warp', 'smooth'],
                    ['noise', 'scale', 'smooth']
                ]
                
                methods = methods_options[aug_idx % len(methods_options)]
                
                for signal in signals:
                    aug_versions = augment_signal_advanced(signal, methods)
                    augmented_signals.append(aug_versions[aug_idx % len(aug_versions)])
                
                # Extract features from augmented signals
                augmented_features = extract_all_features_from_subject(augmented_signals)
                
                # Perturb spatiotemporal
                step_w = data.get('step_width', np.nan)
                cycle_d = data.get('cycle_duration', np.nan)
                vel = data.get('velocity', np.nan)
                
                if not np.isnan(step_w):
                    step_w *= np.random.uniform(0.9, 1.1)
                if not np.isnan(cycle_d):
                    cycle_d *= np.random.uniform(0.9, 1.1)
                if not np.isnan(vel):
                    vel *= np.random.uniform(0.9, 1.1)
                
                augmented_features.extend([step_w, cycle_d, vel])
                
                X_list.append(augmented_features)
                y_list.append(data['label'])
                subject_groups.append(subj_idx)
    
    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    groups = np.array(subject_groups)
    
    print(f"\n  ✅ Dataset:")
    print(f"     Total samples: {len(X)}")
    print(f"     Features per sample: {X.shape[1]}")
    print(f"     Features per signal: 23")
    print(f"     Total signals: 9")
    print(f"     Expected features: {9*23 + 3} = {9*23+3}")
    print(f"     Actual features: {X.shape[1]}")
    print(f"     Healthy: {sum(y=='healthy')}")
    print(f"     K2: {sum(y=='amputee_K2')}")
    print(f"     K3: {sum(y=='amputee_K3')}")
    print(f"     Unique subjects: {len(np.unique(groups))}")
    
    return X, y, groups


# ─────────────────────────────────────────────
# 3. PROPER STACKING WITH NESTED CV
# ─────────────────────────────────────────────
def create_stacking_model():
    """
    Create stacking model (no CV inside - will use nested CV externally).
    """
    # RF base
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # SVM base
    svm_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            probability=True,
            random_state=42,
            class_weight='balanced',
            kernel='rbf',
            C=1.0
        ))
    ])
    
    # Stacking with logistic regression meta-learner
    stacking = StackingClassifier(
        estimators=[
            ('rf', rf_pipeline),
            ('svm', svm_pipeline)
        ],
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        cv=5,  # This is internal to stacking
        stack_method='predict_proba'
    )
    
    return stacking


def nested_cv_with_stacking(X, y, groups):
    """
    Proper nested cross-validation with stacking.
    Outer loop: GroupKFold
    Inner loop: handled by stacking's internal CV
    """
    print("\n" + "=" * 60)
    print("NESTED CROSS-VALIDATION WITH STACKING")
    print("=" * 60)
    
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    all_true = []
    all_pred = []
    all_probs = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create fresh stacking model for each fold
        stacking = create_stacking_model()
        stacking.fit(X_train, y_train)
        
        # Predict
        y_pred = stacking.predict(X_test)
        y_proba = stacking.predict_proba(X_test)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_probs.extend(y_proba)
        
        # Per-fold score
        fold_score = f1_score(y_test, y_pred, average='macro')
        print(f"  Fold {fold+1}: F1-macro = {fold_score:.3f}")
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)
    
    overall_f1 = f1_score(all_true, all_pred, average='macro')
    overall_acc = (all_true == all_pred).mean()
    
    print(f"\n  ✅ Nested CV Results:")
    print(f"     F1-macro (macro avg): {overall_f1:.3f}")
    print(f"     Accuracy: {overall_acc:.3f}")
    
    print("\n  Classification Report (Nested CV):")
    print(classification_report(all_true, all_pred, target_names=['Healthy', 'K2', 'K3']))
    
    return overall_f1, all_true, all_pred, all_probs


# ─────────────────────────────────────────────
# 4. CALIBRATION AND CONFIDENCE
# ─────────────────────────────────────────────
def find_optimal_threshold_from_cv(all_true, all_probs, target_precision=0.85):
    """
    Find optimal confidence threshold using nested CV results.
    """
    # Highest probability for each sample
    max_probs = np.max(all_probs, axis=1)
    
    # Convert probabilities to predictions
    all_pred = np.argmax(all_probs, axis=1)
    
    thresholds = np.arange(0.5, 0.95, 0.01)
    best_threshold = 0.6
    best_precision = 0
    
    for thresh in thresholds:
        confident = max_probs >= thresh
        
        # Make sure there are enough samples
        if np.sum(confident) == 0:
            continue
        
        precision = precision_score(
            all_true[confident],
            all_pred[confident],
            average='macro'
        )
        
        if precision > best_precision and precision >= target_precision:
            best_precision = precision
            best_threshold = thresh
    
    print(f"\n  📊 Optimal confidence threshold: {best_threshold:.2f}")
    print(f"     Precision at threshold: {best_precision:.3f}")
    print(f"     Confident rate: {(max_probs >= best_threshold).mean()*100:.1f}%")
    
    return best_threshold


def train_final_calibrated_model(X, y):
    """
    Train final model with calibration.
    """
    print("\n" + "=" * 60)
    print("TRAINING FINAL CALIBRATED MODEL")
    print("=" * 60)
    
    stacking = create_stacking_model()
    stacking.fit(X, y)
    
    # Calibration for reliable probabilities
    calibrated = CalibratedClassifierCV(stacking, cv=5, method='isotonic')
    calibrated.fit(X, y)
    
    return calibrated


# ─────────────────────────────────────────────
# 5. PREDICTION FUNCTION (FIXED)
# ─────────────────────────────────────────────
def predict_from_subject_data():
    """
    Predict using existing subject data from the loaded Excel files.
    This function replaces the manual feature input.
    """
    global SUBJECT_DATA_CACHE
    
    # Load the trained model
    try:
        with open('gait_classifier_production.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print("\n❌ No model found. Please train first (option 1).")
        return
    
    # Load subject data if not already loaded
    if SUBJECT_DATA_CACHE is None:
        SUBJECT_DATA_CACHE = load_subject_data()
        if SUBJECT_DATA_CACHE is None:
            print("\n❌ Could not load subject data.")
            return
    
    subject_data = SUBJECT_DATA_CACHE
    
    print("\n" + "=" * 60)
    print("PREDICTION WITH CALIBRATED CONFIDENCE")
    print("=" * 60)
    
    # ✅ الخطوة 1: اختاري Subject من البيانات
    print("\n📋 Available Subjects:")
    subject_keys = list(subject_data.keys())
    for i, key in enumerate(subject_keys):
        label = subject_data[key]['label']
        signals_count = len(subject_data[key]['signals'])
        print(f"   {i:>3}. {key:<12} (Label: {label}, Signals: {signals_count})")
    
    try:
        idx = int(input("\n🔢 Enter subject index: "))
        if idx < 0 or idx >= len(subject_keys):
            print("❌ Invalid index!")
            return
        subj_key = subject_keys[idx]
    except ValueError:
        print("❌ Invalid input!")
        return
    
    data = subject_data[subj_key]
    signals = data['signals']
    
    print(f"\n✅ Selected: {subj_key}")
    print(f"   Original label: {data['label']}")
    print(f"   Number of signals: {len(signals)}")
    
    # ✅ الخطوة 2: Feature Extraction
    print("\n📊 Extracting features...")
    
    # Ensure we have 9 signals
    while len(signals) < 9:
        signals.append(np.full(50, np.nan))
    
    features = extract_all_features_from_subject(signals)
    
    # ✅ الخطوة 3: Add spatiotemporal features
    step_width = data.get('step_width', np.nan)
    cycle_duration = data.get('cycle_duration', np.nan)
    velocity = data.get('velocity', np.nan)
    
    features.extend([step_width, cycle_duration, velocity])
    
    # Check feature count
    expected_features = metadata['n_features']
    print(f"   Extracted features: {len(features)}")
    print(f"   Expected features: {expected_features}")
    
    if len(features) != expected_features:
        print(f"   ⚠️ Warning: Feature count mismatch!")
        # Pad or trim as needed
        if len(features) < expected_features:
            features.extend([np.nan] * (expected_features - len(features)))
        else:
            features = features[:expected_features]
    
    # ✅ الخطوة 4: Prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Predict with calibrated probabilities
    probs = model.predict_proba(features_array)[0]
    max_prob = np.max(probs)
    pred_encoded = model.predict(features_array)[0]
    pred = le.inverse_transform([pred_encoded])[0]
    
    threshold = metadata['confidence_threshold']
    is_confident = max_prob >= threshold
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    
    if not is_confident:
        print(f"\n  ⚠️  LOW CONFIDENCE - PREDICTION REJECTED")
        print(f"     Confidence: {max_prob*100:.1f}%")
        print(f"     Threshold: {threshold*100:.1f}%")
        print(f"\n     Original label: {data['label']}")
    else:
        label_map = {
            'healthy': '✅ Healthy (Normal)',
            'amputee_K2': '⚠️ Amputee K2 (Limited Community Ambulator)',
            'amputee_K3': '🟢 Amputee K3 (Full Community Ambulator)'
        }
        print(f"\n  🎯 Diagnosis: {label_map.get(pred, pred)}")
        print(f"  🔒 Confidence: {max_prob*100:.1f}% (Calibrated)")
        print(f"\n     Original label: {data['label']}")
        
        print("\n  📊 Calibrated Probabilities:")
        for cls, prob in zip(le.classes_, probs):
            bar = '█' * int(prob * 40)
            match_marker = "✓" if cls == data['label'] else " "
            print(f"     {match_marker} {cls:>14}: {prob*100:5.1f}%  {bar}")
    
    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Gait Pattern Classifier — PRODUCTION GRADE")
    print("  Classes: Healthy | Amputee K2 | Amputee K3")
    print("=" * 60)
    print("\n  ✅ All Critical Fixes Applied:")
    print("     • Robust feature extraction (no hardcoded indices)")
    print("     • Dynamic percentiles instead of fixed 25/60/75")
    print("     • Advanced augmentation (5 methods)")
    print("     • Proper nested CV with stacking")
    print("     • Calibrated probabilities")
    print("     • FIXED: Proper prediction from existing subject data")
    
    print("\n1. Train model")
    print("2. Predict (using loaded subject data)")
    
    choice = input("\nChoose (1/2): ").strip()
    
    if choice == "1":
        # Build dataset
        X, y, groups = build_feature_matrix_with_advanced_augmentation()
        
        if X is None:
            print("\n❌ Failed to build feature matrix. Please check your Excel files.")
            return
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Nested CV with stacking
        f1_score_val, y_true, y_pred, y_probs = nested_cv_with_stacking(X, y_encoded, groups)
        
        # Find optimal threshold
        threshold = find_optimal_threshold_from_cv(y_true, y_probs, target_precision=0.85)
        
        # Train final calibrated model
        final_model = train_final_calibrated_model(X, y_encoded)
        
        # Save
        with open('gait_classifier_production.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        metadata = {
            'classes': ['healthy', 'amputee_K2', 'amputee_K3'],
            'n_features': X.shape[1],
            'n_samples': len(X),
            'n_subjects': len(np.unique(groups)),
            'nested_cv_f1': f1_score_val,
            'confidence_threshold': threshold,
            'calibration_method': 'isotonic'
        }
        
        with open('model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n  ✅ Model saved: gait_classifier_production.pkl")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy', 'K2', 'K3'],
                    yticklabels=['Healthy', 'K2', 'K3'])
        plt.title('Confusion Matrix - Nested CV')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()
        
    elif choice == "2":
        predict_from_subject_data()
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()