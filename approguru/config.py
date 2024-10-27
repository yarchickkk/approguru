import os
import torch
import torch.nn as nn


# turn off hardware selections, since version is aimed on CPU with default threads number
def select_num_cpu_cores() -> None:
    num_cpu_threads = os.cpu_count()

    if num_cpu_threads is not None:
        # torch.set_num_threads(num_cpu_threads)
        pass
    else:
        print(f"Could not determine the number of CPU threads. Using default settings: {torch.get_num_threads()}.")


def select_torch_device() -> torch.device:
    if False:#torch.cuda.is_available():
        device = torch.device("cuda")
    elif False:#hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        select_num_cpu_cores()
    return device


DEVICE = select_torch_device()
NUM_THREADS = torch.get_num_threads() if DEVICE == torch.device("cpu") else None
INPUT_SIZE = 2
HIDDEN_NEURONS = [8, 16, 16, 16, 8]
ACTIVATION_FUNCTION = nn.LeakyReLU
TARGET_LOSS = 0.005
MAX_ITERATIONS = 10000
MIN_ITERATIONS = 1000
BUFFER_SIZE = 10
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"
