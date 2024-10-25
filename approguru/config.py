import torch
import torch.nn as nn


def select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # optional Apple Silicon GPU support (sucks badly on my device)
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


DEVICE = select_torch_device()
INPUT_SIZE = 2
HIDDEN_NEURONS = [8, 16, 16, 16, 8]
ACTIVATION_FUNCTION = nn.LeakyReLU
TARGET_LOSS = 0.005
MAX_ITERATIONS = 10000
MIN_ITERATIONS = 1000
BUFFER_SIZE = 10


__all__ = []  # no imports to top-level
