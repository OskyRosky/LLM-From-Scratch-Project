import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: Optional[bool] = True) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch to make experiments
    as reproducible as possible.

    Parameters
    ----------
    seed:
        Base integer seed.
    deterministic_torch:
        If True, will try to make PyTorch operations deterministic
        (may slow things down a bit).
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: try to make PyTorch more deterministic
    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # If running on a version where this is not fully supported,
            # we silently ignore it.
            pass