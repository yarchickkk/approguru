import torch
import torch.nn as nn
from datetime import datetime
import plotly.graph_objects as go
from typing import Callable
from .model import MLP
from .config import (
    TARGET_LOSS, MAX_ITERATIONS, MIN_ITERATIONS, BUFFER_SIZE, 
    INPUT_SIZE, DEVICE, RED_BOLD, RESET, FETCHED_DATA_POINTS,
    INITIAL_SEARCH_REGION
)


# @torch.compiler.disable(recursive=True)
def validate_ohlcv_structure(raw_ohlcv: dict) -> bool:
    """
    This function checks wether input data follows expected formatting.

    Args: 
        raw_ohlcv (dict): Raw OHLCV data, obtained from gecoterminal API.

    Returns:
        (bool): whether raw_ohlcv is valid or not. 
    """
    message = f"{RED_BOLD}(guru){RESET} Skipping graph processing. Seems like input dictionary isn't structured properly.\n"
    
    try:
        pool_address = raw_ohlcv["meta"]["base"]["address"]
    except (KeyError, TypeError):
        print(f"{message} ----> Make sure it has pool address accessed via ['meta']['base']['address'].")
        return False
    
    try:
        ohlcv_list = raw_ohlcv["data"]["attributes"]["ohlcv_list"]
        if isinstance(ohlcv_list, list) and all(isinstance(sublist, list) for sublist in ohlcv_list):
            if all((len(sublist) == 6 and all(isinstance(elem, (int, float)) for elem in sublist)) for sublist in ohlcv_list):
                if len(ohlcv_list) == FETCHED_DATA_POINTS:
                    return True
                else:
                    print(f"{message} ----> Make sure OHLCV list length is {FETCHED_DATA_POINTS}, bit not {len(ohlcv_list)}.\n")
                    return False
    except (KeyError, TypeError):
        print(f"{message} ----> Make sure it has OHLCV list accessed via ['data']['attributes']['ohlcv_list'].\n"
              f" ----> Pool address: '{pool_address}'.")
        return False


# @torch.compile
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
    
    return [row.view(-1, 1) for row in out] + [data]


# @torch.compile
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


# @torch.compile
def get_feature_gradients(model: MLP, X: torch.Tensor) -> torch.Tensor:
    """
    Backpropagates to the original features through the model and polynomial transformation.
    Clears model's gradients after exedcution.
    """
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

# @torch.compile
def get_feature_gradients_v2(model: MLP, X_normalized: torch.Tensor) -> torch.Tensor:
    
    # input transformation
    Xn = X_normalized
    Xp = polynomial_features(Xn, model.ipt_size)
    # model output
    Yn = model(Xp)

    grads = []
    for i in range(1, Xn.shape[0]):
        l, r = Yn[i - 1], Yn[i]

        if r > l:
            grads.append(1.0)
        else:
            grads.append(-1.0)
    
    if Yn[-1] > Yn[-2]:
        grads.append(1.0)
    else:
        grads.append(-1.0)
    
    grads = torch.tensor(grads).view(-1, 1)
    return grads

# @torch.compiler.disable(recursive=True)
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

    return torch.tensor(iter), loss.detach()


# @torch.compiler.disable(recursive=True)
def adjust_region_start(X_normalized_gradients: torch.Tensor) -> int:
    """
    This function checks whether the initial portion of the search region shows a 
    downward trend (fall) at the start of the graph. If a fall is detected, it expands 
    the search region to the left to include the entire negative trend.
    """
    grads = X_normalized_gradients

    # take default amount of points we're performing search on (right part of grads)
    start_ptr = FETCHED_DATA_POINTS - INITIAL_SEARCH_REGION 
    
    # step left is possible, and both current and previous points have a negative gradient
    while (start_ptr - 1) >= 0 and grads[start_ptr] <= 0.0 and grads[start_ptr - 1] <= 0.0:
        start_ptr -= 1
    return start_ptr


# @torch.compiler.disable(recursive=True)
@torch.no_grad()
def find_max_negative_slope(X_normalized_gradients: torch.Tensor, Y_original: torch.Tensor, start_pointer: int) -> list[torch.Tensor]:
    """
    Identifies the extrema of the model's approximation, detects the steepest negative slope, 
    and returns it's ratio after applying a buffer search correction.
    """

    # cut unused points on the left
    grads, Y = X_normalized_gradients[start_pointer:], Y_original[start_pointer:]
    max_idx = grads.shape[0] - 1  # most index in gradients array

    sign_changes1 = torch.where(grads[:-1] * grads[1:] <= 0)[0]  # the indicies after which the sign changes in gradients tensor
    
    # more accurate way to point at extremums
    sign_changes = []
    for i in range(1, grads.shape[0]):
        l, r = i - 1, i
        l_grad, r_grad = grads[l], grads[r]
        l_val, r_val = Y[l], Y[r]
        
        if l_grad * r_grad <= 0:  # detect a sign change

            if l_grad <= 0 and r_grad > 0: # minimum
                change = l if l_val <= r_val else r

            elif l_grad > 0 and r_grad <= 0: # maximum
                change = l if l_val > r_val else r
            
            sign_changes.append(change)

    extremums = list(sign_changes)
    if sign_changes[0] != 0:  # mark starting index as extremum if it's not already
        extremums = [0] + extremums
    
    if sign_changes[-1] != torch.tensor(max_idx):  # same for last index
        extremums = extremums + [max_idx]
    extremums = torch.tensor(extremums)
    # extremums = torch.cat([torch.tensor([0], device=DEVICE), sign_changes, torch.tensor([max_idx], device=DEVICE)])  # add first and last indicies

    max_negative_slope = torch.tensor(float("-inf"))
    min_val_idx, max_val_idx = None, None

    for s, e in zip(extremums[:-1], extremums[1:]):
        # exclude intervals of growth
        if (Y[e] / Y[s] - 1) > 0.0:
            continue

        s = torch.clamp(s - 0, min=0)  # include left buffer
        e = torch.clamp(e + 0, max=max_idx + 1)  # include right buffer

        min_val_i, min_val_idx_i = torch.min(Y[s:e+1], dim=0)  # min value an it's local index
        max_val_i, max_val_idx_i = torch.max(Y[s:e+1], dim=0)  # max value an it's local index

        max_negative_slope_i = max_val_i / min_val_i - 1.0

        if max_negative_slope_i > max_negative_slope:
            max_negative_slope = max_negative_slope_i
            max_val_idx = max_val_idx_i + s
            min_val_idx = min_val_idx_i + s

    if max_negative_slope == torch.tensor(float("inf")):
        print(f"{RED_BOLD}(guru){RESET} No falls were found. Seems like the graph constantly grows over given interval.\n"
              " ----> find_max_negative_slope() returned [None, None, None, None].")
        return None, None, None, None

    # adjust returned indicies to align with the original tensor
    return max_negative_slope, extremums + start_pointer, min_val_idx + start_pointer, max_val_idx + start_pointer


def get_open_close_bound(ohlcv_list: dict, candle_index: int, func: Callable) -> None:
    """
    Get the maximum or minimum of the open and close values from a given candle.
    """
    assert func in (max, min), f"{RED_BOLD}(approguru){RESET} 'func' argument must be either 'max' or 'min' function."

    data = ohlcv_list[candle_index].view(-1)
    oc = [data[1], data[2]] # open and close  ATTENTION: USING HIGH AS TARGETS!!!
    
    return func(oc)


# @torch.compiler.disable(recursive=True)
def timestamps_to_dates(timestamps: torch.Tensor) -> list[str]:
    """
    Convert a tensor of timestamps to a tensor of date strings.
    """
    format = "%d-%m-%Y %H:%M:%S"
    date_strings = [datetime.fromtimestamp(ts.item()).strftime(format) for ts in timestamps]
    return date_strings


# @torch.compiler.disable(recursive=False)
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
