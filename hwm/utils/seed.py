import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    """Seed python, numpy and torch (CPU + CUDA). Returns a fresh numpy Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
