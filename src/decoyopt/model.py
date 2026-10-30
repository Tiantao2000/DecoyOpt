"""Independent binary/ranking model and native CLEAN EC encoder."""
from __future__ import annotations
import torch
from torch import nn

MODEL_POLICY = "independent_binary_native_clean_v1"


class BinaryModel(nn.Module):
    """No dropout. No parameters shared with ECModel."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.binary = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.binary(x).squeeze(-1)


class ECModel(nn.Module):
    """CLEAN.LayerNormNet; only its fixed 1280 input width is made configurable.

    Constructor order, dropout placement and forward match the original CLEAN LayerNormNet.
    """
    def __init__(self, input_dim: int, device=None, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(512, dtype=dtype, device=device)
        self.fc2 = nn.Linear(512, 512, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(512, dtype=dtype, device=device)
        self.fc3 = nn.Linear(512, 128, dtype=dtype, device=device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.dropout(self.ln1(self.fc1(x))))
        x = torch.relu(self.dropout(self.ln2(self.fc2(x))))
        return self.fc3(x)
