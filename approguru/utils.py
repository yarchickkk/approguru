import torch
import torch.nn as nn
from datetime import datetime
import plotly.graph_objects as go
from .model import MLP
from .config import (
    TARGET_LOSS,
    MAX_ITERATIONS,
    MIN_ITERATIONS,
    BUFFER_SIZE,
    INPUT_SIZE,
    DEVICE
)


def validate_ohlcv_structure(raw_ohlcv: dict) -> bool:
    """
    This function checks wether input data follows expected formatting.

    Args: 
        raw_ohlcv (dict): Raw OHLCV data, obtained from gecoterminal API.

    Returns:
        (bool): whether raw_ohlcv is valid or not. 
    """
    red_bold, reset = "\033[1;31m", "\033[0m"
    message = f"{red_bold}(guru){reset} Skipping graph processing. Seems like input dictionary isn't structured properly.\n"
    
    try:
        pool_address = raw_ohlcv["meta"]["base"]["address"]
    except KeyError:
        print(f"{message} ----> Make sure it has pool address accessed via ['meta']['base']['address'].")
        return False
    
    try:
        ohlcv_list = raw_ohlcv["data"]["attributes"]["ohlcv_list"]
        if isinstance(ohlcv_list, list) and all(isinstance(sublist, list) for sublist in ohlcv_list):
            if all(len(sublist) == 5 and all(isinstance(elem, (int, float)) for elem in sublist) for sublist in ohlcv_list):
                raise KeyError
    except (KeyError, TypeError):
        print(f"{message} ----> Make sure it has OHLCV list accessed via ['data']['attributes']['ohlcv_list'].\n"
              f" ----> Pool address: '{pool_address}'.")
        return False


# @torch.compile()
def preprocess_data(raw_ohlcv: dict, features: str = "timestamp", targets: str = "close") -> list[torch.Tensor]:
    """
    This function transforms raw OHLCV data into a format that can be fed into a model. Applies min-max normalization.

    Args:
        raw_ohlcv (dict): Raw OHLCV data, obtained from gecoterminal API.
        features (str)  : Specifies what part of OHLCV is treated as X.
        targets (str)   : Specifies what part of OHLCV is treated as Y.
    
    Returns:
        [X, Y, X_normalized, Y_normalized]: list of column tensors of shape (-1, 1).
    """
    
    stoi = {"timestamp": 0,  # string to index mappings
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5}

    data = torch.tensor(raw_ohlcv['data']['attributes']['ohlcv_list'], dtype=torch.float32, device=DEVICE)  # list -> torch.tensor
    data = torch.flip(data, dims=[0])  # [present -> past] -> [past -> present]

    out = [data[:, stoi[features]], data[:, stoi[targets]]]  # X, Y

    for set in out[:2]:
        setn = (set - set.min()) / (set.max() - set.min())  # min-max normalization
        out.append(setn) 
    
    return [row.view(-1, 1) for row in out]


# @torch.compile()
def polynomial_features(X_original: torch.Tensor, max_pow: int) -> torch.Tensor:
    """
    This function takes a tensor of shape (-1, 1) and returns a new tensor 
    where each column corresponds to the input features raised to the 
    powers from 1 to 'max_pow'.
    """
    x = X_original
    assert x.dim() == 2 and x.shape[-1] == 1, f"Input tensor must be of shape (-1, 1). Got {tuple(x.shape)} instead."
    powers = torch.arange(1, max_pow + 1, device=DEVICE)
    return torch.pow(x, powers)


# @torch.compile()
def get_original_feature_gradients(model: MLP, X_original: torch.Tensor) -> torch.Tensor:
    """
    Backpropagates to the original features through the model and polynomial transformation.
    Clears model's gradients after exedcution.
    """
    X = X_original
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


# @torch.compile()
def train_mlp(model: MLP, X_polynomial: torch.Tensor, Y_normalized: torch.Tensor) -> list[torch.Tensor]:
    """
    Trains the provided model using the given training datasets until 
    either a target performance metric is achieved 
    or a specified number of optimization steps are completed.
    """
    Xtr, Ytr = X_polynomial, Y_normalized
    mse_loss = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss, target_loss = torch.tensor(float("inf"), device=DEVICE), TARGET_LOSS
    iter, max_iters, min_iters = 0, MAX_ITERATIONS, MIN_ITERATIONS
    
    while (loss.item() > target_loss and iter < max_iters) or (iter < min_iters):
        optimizer.zero_grad(set_to_none=True)
        preds = model(Xtr)
        loss = mse_loss(preds, Ytr)
        loss.backward()
        optimizer.step()
        iter += 1

    return iter, loss


# @torch.compile()
# @torch.no_grad()
def find_max_negative_slope(model: MLP, X_normalized: torch.Tensor, Y_original: torch.Tensor) -> list[torch.Tensor]:
    """
    Identifies the extrema of the model's approximation, detects the steepest negative slope, 
    and returns it's ratio after applying a buffer search correction.
    """
    Xn, Y = X_normalized, Y_original
    
    grads = get_original_feature_gradients(model, Xn)
    max_idx = grads.shape[0] - 1  # most index in gradients array

    sign_changes = torch.where(grads[:-1] * grads[1:] <= 0)[0]  # the indicies after which the sign changes in gradients tensor
    extremums = torch.cat([torch.tensor([0], device=DEVICE), sign_changes, torch.tensor([max_idx], device=DEVICE)])  # add first and last indicies

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

    return max_negative_slope, extremums, min_val_idx, max_val_idx 


def timestamps_to_dates(timestamps: torch.Tensor) -> list[str]:
    """
    Convert a tensor of timestamps to a tensor of date strings.
    """
    format = "%d-%m-%Y %H:%M:%S"
    date_strings = [datetime.fromtimestamp(ts.item()).strftime(format) for ts in timestamps]
    return date_strings

# @torch.compile()
# @torch.no_grad()
def visualize_model(model: MLP, X_original: torch.Tensor, Y_original: torch.Tensor, X_normalized: torch.Tensor, 
                    Y_normalized: torch.Tensor, extremums: torch.Tensor, min_val_idx: torch.Tensor, max_val_idx: torch.Tensor) -> None:
    X, Y, Xn, Yn = X_original, Y_original, X_normalized, Y_normalized

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
        x0=Xnplot[min_val_idx.item()],
        x1=Xnplot[max_val_idx.item()],
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
