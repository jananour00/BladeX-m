import torch
import torch.nn as nn
import math

class FastTemporalTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, dropout=0.3, seq_len=30):
        super().__init__()
        self.input_dim
