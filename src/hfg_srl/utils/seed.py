"""Seed management for reproducibility."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(device_str: str = "auto") -> str:
    """Get computation device.

    Args:
        device_str: "auto", "cpu", or "cuda".

    Returns:
        Device string.
    """
    if device_str == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"
    return device_str
