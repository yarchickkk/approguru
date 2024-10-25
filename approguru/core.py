import torch
import torch.nn as nn
from .model import MLP
from .utils import (
    validate_ohlcv_structure,
    preprocess_data,
    polynomial_features,
    train_mlp,
    find_max_negative_slope,
    visualize_model
)
from .config import (
    INPUT_SIZE, HIDDEN_NEURONS, ACTIVATION_FUNCTION, DEVICE
)


def magic_function(data: dict, seed: int = 13, log_training: bool = True, visualize_graph: bool = True) -> float:
    if validate_ohlcv_structure(data) is False:
        return None
    
    X, Y, Xn, Yn = preprocess_data(data)
    Xp = polynomial_features(Xn, INPUT_SIZE)

    torch.manual_seed(seed)  # stable weight initialization
    model = MLP(INPUT_SIZE, HIDDEN_NEURONS, ACTIVATION_FUNCTION)
    model.to(DEVICE)
    iters, loss = train_mlp(model, Xp, Yn)
    if log_training is True:
        print(f"{iters} iterations, {loss} loss.")
    
    slope, extremums, min_val_idx, max_val_idx = find_max_negative_slope(model, Xn, Y)
    
    if visualize_graph is True:
        visualize_model(model, X, Y, Xn, Yn, extremums, min_val_idx, max_val_idx)
    return slope.item()


class MaxFallFinder(nn.Module):
    
    def __init__(self, seed: int = 13) -> None:
        super().__init__()
        self.seed = seed
    
    def forward(self, ohlcv_data: dict) -> None:
        if validate_ohlcv_structure(ohlcv_data) is False:  # data validation
            return None
       
        self.X, self.Y, self.Xnorm, self.Ynorm = preprocess_data(ohlcv_data)  # data preprocessing
        self.Xpoly = polynomial_features(self.Xnorm, INPUT_SIZE)
        print("Prepared data!")
        torch.manual_seed(self.seed)  # model initialization
        self.mlp = MLP(
            ipt_size=INPUT_SIZE,
            hidden_ns=HIDDEN_NEURONS,
            act_layer=ACTIVATION_FUNCTION
        )
        self.mlp.to(DEVICE)
        print("Initialized model!")
        self.steps_made, self.achieved_loss = train_mlp(  # model training
            model=self.mlp,
            X_polynomial=self.Xpoly,
            Y_normalized=self.Ynorm
        )
        print(f"Trained model:\n{self.steps_made} steps, {self.achieved_loss} loss")
        self.max_fall, self.extremums, self.min_val_idx, self.min_val_idx = find_max_negative_slope( 
            model=self.mlp,
            X_normalized=self.Xnorm,
            Y_original=self.Y
        )
        print("Got the data!")


__all__ =  []  # no imports to top-level
