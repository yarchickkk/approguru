import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Type


with open("data/MARS_data.json", "r") as file:
    mars_data = json.load(file)


INPUT_SIZE = 2
HIDDEN_NEURONS = [8, 16, 16, 16, 8]
TARGET_LOSS = 0.005
MAX_ITERATIONS = 10000
MIN_ITERATIONS = 1000
BUFFER_SIZE = 10


def preprocess_data(raw_ohlcv: dict, features: str = "timestamp", targets: str = "close", epsilon: float = 1e-8) -> list[torch.tensor]:
    """
    This function transforms raw OHLCV data into a format that can be fed into a model. 

    Returns:
    1. The raw feature (X) and target (Y) tensors.
    2. The normalized feature (Xn) and target (Yn) tensors, scaled using min-max normalization with optional epsilon adjustment for zero values.
    """
    stoi = {"timestamp": 0,  # string to index mappings
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5}
    
    data = torch.tensor(raw_ohlcv['data']['attributes']['ohlcv_list'], dtype=torch.float32)  # list -> torch.tensor
    data = torch.flip(data, dims=[0])  # [present -> past] -> [past -> present]

    out = [data[:, stoi[features]], data[:, stoi[targets]]]  # X, Y

    for set in out[:2]:
        setn = (set - set.min()) / (set.max() - set.min())  # Xn, Yn
        setn = setn + (setn == 0).float() * epsilon  # epsilon adjustment
        out.append(setn) 
    
    return [row.view(-1, 1) for row in out]  # [X, Y, Xn, Yn] as (-1, 1) column tensors


def polynomial_features(x: torch.tensor, max_pow: int) -> torch.tensor:
    """
    This function takes a tensor of shape (-1, 1) and returns a new tensor 
    where each column corresponds to the input features raised to the 
    powers from 1 to 'max_pow'.
    """
    assert x.dim() == 2 and x.shape[-1] == 1, f"Input tensor must be of shape (-1, 1). Got {tuple(x.shape)} instead."
    powers = torch.arange(1, max_pow + 1)
    return torch.pow(x, powers)


class MLP(nn.Module):

    def __init__(self, ipt_size: int, hidden_ns: list, act_layer: Type[nn.Module] = nn.ELU) -> None:
        super().__init__()
        self.ipt_size = ipt_size

        layers = [nn.Linear(self.ipt_size, hidden_ns[0]), act_layer()]  # input layer + relu
        for n0, n1 in zip(hidden_ns[:-1], hidden_ns[1:]):  # hidden layers + relus
            layers.extend([nn.Linear(n0, n1), act_layer()])
        layers.append(nn.Linear(hidden_ns[-1], 1))  # output layer

        self.layers = nn.Sequential(*layers)  # stack layers
    
    def forward(self, x_polynomial: torch.tensor) -> torch.tensor:
        return self.layers(x_polynomial)
    
    def set_gradients_to_none(self) -> None:
        for param in self.parameters():
            param.grad = None


def get_original_feature_gradients(model: MLP, orig_features: torch.tensor) -> torch.tensor:
    """
    Backpropagates to the original features through the model and polynomial transformation.
    Clears model's gradients after exedcution.
    """
    X = orig_features
    model.set_gradients_to_none()  # clean gradients
    X.requires_grad_(True) # include in computation

    Xpol = polynomial_features(X, model.ipt_size)
    approximations = model(Xpol)
    approximations_sum = approximations.sum()  # single value to backward from

    approximations_sum.backward()
    grads_wrt_orig_features = X.grad

    X.grad = None  # clean gradient
    X.requires_grad_(False)  # exclude from computation
    model.set_gradients_to_none()  # clean gradients

    return grads_wrt_orig_features


def train_mlp(model: MLP, Xtr: torch.tensor, Ytr: torch.tensor, verbosity: bool = False) -> None:
    """
    Trains the provided model using the given training datasets until 
    either a target performance metric is achieved 
    or a specified number of optimization steps are completed.
    """
    mse_loss = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss, target_loss = torch.tensor(float("inf")), TARGET_LOSS
    iter, max_iters, min_iters = 0, MAX_ITERATIONS, MIN_ITERATIONS
    
    while (loss.item() > target_loss and iter < max_iters) or (iter < min_iters):
        optimizer.zero_grad(set_to_none=True)
        preds = model(Xtr)
        loss = mse_loss(preds, Ytr)
        loss.backward()
        optimizer.step()
        iter += 1

    if verbosity is True:
        print(f"{iter} iterations, {loss} loss")


def find_max_negative_slope(model: MLP, orig_features: torch.tensor, orig_targets: torch.tensor) -> torch.tensor:
    """
    Identifies the extrema of the model's approximation, detects the steepest negative slope, 
    and returns it's ratio after applying a buffer search correction.
    """
    X, Y = orig_features, orig_targets
    
    grads = get_original_feature_gradients(model, X)
    max_idx = grads.shape[0] - 1  # most index in gradients array

    sign_changes = torch.where(grads[:-1] * grads[1:] <= 0)[0]  # the indicies after which the sign changes in gradients tensor
    extremums = torch.cat([torch.tensor([0]), sign_changes, torch.tensor([max_idx])])  # add first and last indicies

    ratios = Y[extremums[1:]] / Y[extremums[:-1]] - 1  # get fall ratios on each interval
    max_fall_idx = ratios.argmin()  # find interval with the most negative fall
    s, e = extremums[max_fall_idx], extremums[max_fall_idx + 1]  # restore it's boundary indicies

    buff_size = BUFFER_SIZE
    s = torch.clamp(s - buff_size, min=0)  # include left buffer
    e = torch.clamp(e + buff_size, max=max_idx + 1)  # include right buffer

    min_val, max_val = torch.aminmax(Y[s:e])
    max_negative_slope = min_val / max_val - 1.0

    return max_negative_slope  # torch.tensor of a single element


X, Y, Xn, Yn = preprocess_data(mars_data, epsilon=0.0)

input_size = INPUT_SIZE
hidden_neurons = HIDDEN_NEURONS

model = MLP(input_size, hidden_neurons, nn.LeakyReLU)
Xp = polynomial_features(Xn, input_size)

train_mlp(model, Xp, Yn, verbosity=True)

answer = find_max_negative_slope(model, Xn, Y).item()
print(answer)

# Useless
preds = model(Xp)
grads = get_original_feature_gradients(model, Xn)
