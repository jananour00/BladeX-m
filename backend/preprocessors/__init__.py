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

__all__ = [
    'FatiguePreprocessor', 'FEATURES_FATIGUE', 'FEATURES_INJURY',
    'QualityPreprocessor', 'PERFORMANCE_LEVELS',
]
