"""
Preprocessors Package
=====================
Contains model-specific preprocessing logic.

Each preprocessor is isolated because each model was trained on
completely different datasets with different feature spaces:

  fatigue_preprocessor.py  → Fatigue + Injury models (Paralympic runners CSV)
  quality_preprocessor.py  → Quality of Movement + Coach Assistant
"""
from .fatigue_preprocessor import FatiguePreprocessor, FEATURES_FATIGUE, FEATURES_INJURY
from .quality_preprocessor import QualityPreprocessor, PERFORMANCE_LEVELS
from .bayes_preprocessor import load_scalers, get_feature_columns, FEATURES_BAYES
from .cgnn_preprocessor import load_cgnn_scalers, get_cgnn_feature_columns
from .qom_preprocessor import QoMPreprocessor, QOM_FEATURES
from .temporal_fatigue_preprocessor import TemporalFatiguePreprocessor, FEATURES_TEMPORAL_FATIGUE, SEQ_LEN

__all__ = [
    'FatiguePreprocessor', 'FEATURES_FATIGUE', 'FEATURES_INJURY',
    'QualityPreprocessor', 'PERFORMANCE_LEVELS',
    'load_scalers', 'get_feature_columns', 'FEATURES_BAYES',
    'load_cgnn_scalers', 'get_cgnn_feature_columns',
    'QoMPreprocessor', 'QOM_FEATURES',
    'TemporalFatiguePreprocessor', 'FEATURES_TEMPORAL_FATIGUE', 'SEQ_LEN'
]
