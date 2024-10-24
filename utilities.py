import torch
import torch.nn as nn
from typing import Type
from datetime import datetime
import plotly.graph_objects as go
from globals import *


def preprocess_data(raw_ohlcv: dict, features: str = "timestamp", targets: str = "close") -> list[torch.tensor]:
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
    """
    Casual Multi Layer Preseptron.
    """
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


def find_max_negative_slope(model: MLP, norm_features: torch.tensor, orig_targets: torch.tensor, save_visualized_data: bool = False) -> torch.tensor:
    """
    Identifies the extrema of the model's approximation, detects the steepest negative slope, 
    and returns it's ratio after applying a buffer search correction.
    """
    Xn, Y = norm_features, orig_targets
    
    grads = get_original_feature_gradients(model, Xn)
    max_idx = grads.shape[0] - 1  # most index in gradients array

    sign_changes = torch.where(grads[:-1] * grads[1:] <= 0)[0]  # the indicies after which the sign changes in gradients tensor
    extremums = torch.cat([torch.tensor([0]), sign_changes, torch.tensor([max_idx])])  # add first and last indicies

    ratios = Y[extremums[1:]] / Y[extremums[:-1]] - 1  # get fall ratios on each interval
    max_fall_idx = ratios.argmin()  # find interval with the most negative fall
    s, e = extremums[max_fall_idx], extremums[max_fall_idx + 1]  # restore it's boundary indicies

    buff_size = BUFFER_SIZE
    s = torch.clamp(s - buff_size, min=0)  # include left buffer
    e = torch.clamp(e + buff_size, max=max_idx + 1)  # include right buffer
    
    min_val, min_val_idx = torch.min(Y[s:e], dim=0)  # min value an it's local index
    max_val, max_val_idx = torch.max(Y[s:e], dim=0)  # max value an it's local index
    min_val_idx, max_val_idx = min_val_idx + s, max_val_idx + s  # global indicies

    max_negative_slope = min_val / max_val - 1.0

    if save_visualized_data is True:
        model.visualized_data = [  # Xn, Y, extremums, fall_start, fall_end
            norm_features, orig_targets, 
            extremums, min_val_idx, max_val_idx 
        ]

    return max_negative_slope  # torch.tensor of a single element


def timestamps_to_dates(timestamps: torch.tensor):
    """
    Convert a tensor of timestamps to a tensor of date strings.
    """
    format = "%d-%m-%Y %H:%M:%S"
    date_strings = [datetime.fromtimestamp(ts.item()).strftime(format) for ts in timestamps]
    return date_strings


def visualize_model(model: MLP, X_original: torch.tensor, Y_normalized: torch.tensor) -> None:
    X, Yn = X_original, Y_normalized
    try:
        model.visualized_data
    except AttributeError as e:
        print("'MLP' object has no attribute 'visualized_data'")

    Xn, Y, extremums, mini, maxi = model.visualized_data  # Unpack data from find_max_negative_slope() 
    preds = model(polynomial_features(Xn, INPUT_SIZE))  # Forward model

    Xnplot = Xn.view(-1).numpy()
    Ynplot = Yn.view(-1).numpy()
    dates = timestamps_to_dates(X.view(-1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(  # approximated function
        x=Xnplot, 
        y=Ynplot, 
        mode='lines', 
        name='Price', 
        line=dict(width=1, color='#1f77b4'),
        text=dates, 
        customdata=Y.view(-1).numpy(),
        hovertemplate="%{text}<br>%{customdata:.14f}$<extra></extra>"
        ))

    fig.add_trace(go.Scatter(  # approximation
        x=Xnplot, 
        y=preds.detach().view(-1).numpy(), 
        mode='lines', 
        name='Approximation',
        line=dict(width=2, color='#ff7f0e'),
        hoverinfo='none'
        ))

    for i in extremums:  # extremums
        x_value = Xnplot[i]
        fig.add_shape(
            type='line',
            x0=x_value, x1=x_value,
            y0=min(Ynplot), y1=max(Ynplot),
            line=dict(color='#2ca02c', width=2, dash="dash"),
            name="Exteremum",
            showlegend=(True if i == 0 else False)
        )

    fig.add_shape(  # fall region
        type='rect',
        x0=Xnplot[mini.item()],
        x1=Xnplot[maxi.item()],
        y0=min(Ynplot),
        y1=max(Ynplot),
        fillcolor='#d62728',
        opacity=0.25,
        line_width=0,
        showlegend=True,
        name="Fall Region"
    )

    fig.update_layout(  # display properties
        margin=dict(l=0, r=0, t=0, b=0),
        template='simple_white',
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="rgba(211, 211, 211, 0.4)",
            gridwidth=0.05
        )
    )
    fig.show()


def magic_function(data: dict, seed: int = 13, log_training: bool = True, visualize_graph: bool = True) -> float:
    X, Y, Xn, Yn = preprocess_data(data)
    Xp = polynomial_features(Xn, INPUT_SIZE)

    torch.manual_seed(seed)  # stable weight initialization
    model = MLP(INPUT_SIZE, HIDDEN_NEURONS, ACTIVATION_FUNCTION)
    train_mlp(model, Xp, Yn, verbosity=log_training)
    slope = find_max_negative_slope(model, Xn, Y, save_visualized_data=visualize_graph)
    
    if visualize_graph is True:
        visualize_model(model, X, Yn)
    return slope.item()
