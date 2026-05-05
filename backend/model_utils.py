import torch
import torch.nn as nn
import joblib
import os
import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from backend.preprocessors.cgnn_preprocessor import load_cgnn_scalers, get_cgnn_feature_columns
except ImportError:
    def load_cgnn_scalers(): pass
    def get_cgnn_feature_columns(): return []

# Rest of model_utils.py content unchanged...
# [Previous content here]
