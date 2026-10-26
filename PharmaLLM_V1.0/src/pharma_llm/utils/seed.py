from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Set random seeds for supported libraries."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from transformers import set_seed as transformers_set_seed

        transformers_set_seed(seed)
    except ImportError:
        pass

