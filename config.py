"""
config.py — Shared constants and device setup.
"""

import torch
import numpy as np

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 128
LR         = 1e-3