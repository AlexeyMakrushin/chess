"""
MLP model for chess position evaluation.
"""

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError("PyTorch is required to use model.py") from exc


class ChessEvaluatorMLP(nn.Module):
    """Simple MLP: 768 → 256 → 64 → 1 (tanh output)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
