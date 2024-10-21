# Imports
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Type


# Data Loading
with open("data/trump_data.json", "r") as file:
    trump_data = json.load(file)

with open("data/pepe_data.json", "r") as file:
    pepe_data = json.load(file)

with open("data/spx_data.json", "r") as file:
    spx_data = json.load(file)

with open("data/mstr_data.json", "r") as file:
    mstr_data = json.load(file)


# Modules
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
    
    def get_original_feature_gradients(self, orig_features: torch.tensor) -> torch.tensor:
        self.set_gradients_to_none()  # clean gradients
        orig_features.requires_grad_(True) # include in computation

        Xpol = polynomial_features(orig_features, self.ipt_size)
        approximations = self(Xpol)
        approximations_sum = approximations.sum()  # single value to backward from

        approximations_sum.backward()
        grads_wrt_orig_features = orig_features.grad

        orig_features.grad = None  # clean gradient
        orig_features.requires_grad_(False)  # exclude from computation
        self.set_gradients_to_none()  # clean gradients

        return grads_wrt_orig_features
    
    def set_gradients_to_none(self) -> None:
        for param in self.parameters():
            param.grad = None


# Data preprocessing
X, Y, Xn, Yn = preprocess_data(mstr_data, epsilon=0.0)


# Hyperparameters and model initialization
input_size = 2
hidden_neurons = [16, 32, 16]  # [4, 8, 4]  # [100, 200, 100]

model = MLP(input_size, hidden_neurons, nn.LeakyReLU)
Xp = polynomial_features(Xn, input_size)

mse_loss = nn.MSELoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Training loop
num_iters = 100
for i in range(num_iters):
    optimizer.zero_grad(set_to_none=True)
    preds = model(Xp)
    loss = mse_loss(preds, Yn)
    print(loss)
    loss.backward()
    optimizer.step()


# Backprop to features
grads = model.get_original_feature_gradients(Xn)


# Getting minimums
fig, ax = plt.subplots(figsize=(50, 10))

ax.plot(Xn, Yn)
ax.plot(Xn, preds.detach())

extremums = [0]
for i, (prev, next) in enumerate(zip(grads, grads[1:])):
    if prev <= 0 and next > 0:
        ax.axvline(x=Xn[i], color='green', linestyle='--', linewidth=1)
        extremums.append(i)
    elif prev > 0 and next <= 0:
        ax.axvline(x=Xn[i], color='red', linestyle='--', linewidth=1)
        extremums.append(i)
extremums.append(287)

ax.margins(x=0);


# Getting max fall
for i, (prev, next) in enumerate(zip(extremums, extremums[1:])):
    print(f"{i}: {(Y[next] / Y[prev] - 1).item():+.10f} %")  # AFTER_DEBUG: devision by zero can occur here! 
