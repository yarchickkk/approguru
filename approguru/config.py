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
HIDDEN_NEURONS = [64, 64, 64, 64, 64, 64]  #[2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2]  #[x*4 for x in [8, 16, 32, 32, 16, 8]]
ACTIVATION_FUNCTION = nn.LeakyReLU
TARGET_LOSS = 0.0008
MAX_ITERATIONS = 40000
MIN_ITERATIONS = 20000
BUFFER_SIZE = 0
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"
FETCHED_DATA_POINTS = 400  # candles requested from geckoterminal API
INITIAL_SEARCH_REGION = 350  # candles considered for search by deafault (can be extended during execution)
SEED = 123456
TARGETS_FIT = "close"
TARGETS_SEARCH = ["open", "close"]
GROWTH_PERCENT = 5.0

