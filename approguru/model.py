import torch
import torch.nn as nn
from typing import Type


class MLP(nn.Module):
    """
    Casual Multi Layer Preseptron.
    """
    def __init__(self, ipt_size: int, hidden_ns: list, act_layer: Type[nn.Module] = nn.LeakyReLU) -> None:
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

