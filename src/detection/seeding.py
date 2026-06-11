# -*- coding: utf-8 -*-
"""
Deterministic seeding helpers

Centralises the random-state setup that every experiment shares so that runs are
reproducible across Python, NumPy and PyTorch, including the CUDA backend
"""

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED):
    """
    Seed all relevant random number generators and enforce deterministic cuDNN
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
