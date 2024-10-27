import torch
import torch.nn as nn
import torch.multiprocessing as mp
from .model import MLP
from .utils import (
    validate_ohlcv_structure,
    preprocess_data,
    polynomial_features,
    train_mlp,
    get_feature_gradients,
    find_max_negative_slope,
    visualize_model
)
from .config import (
    INPUT_SIZE, HIDDEN_NEURONS, ACTIVATION_FUNCTION, 
    DEVICE, RED_BOLD, RESET
)




class MaxFallFinder(nn.Module):
    
    def __init__(self, seed: int = 13) -> None:
        super().__init__()
        self.seed = seed

    def forward(self, ohlcv_data: dict) -> None:
        if validate_ohlcv_structure(ohlcv_data) is False:  # data validation
            return None
       
        self.X, self.Y, self.Xnorm, self.Ynorm = preprocess_data(ohlcv_data)  # data preprocessing
        self.Xpoly = polynomial_features(self.Xnorm, INPUT_SIZE)

        torch.manual_seed(self.seed)  # model initialization
        self.mlp = MLP(
            ipt_size=INPUT_SIZE,
            hidden_ns=HIDDEN_NEURONS,
            act_layer=ACTIVATION_FUNCTION
        )
        # torch.compile(self.mlp)
        self.mlp.to(DEVICE)

        self.steps_made, self.achieved_loss = train_mlp(  # model training
            model=self.mlp,
            X_polynomial=self.Xpoly,
            Y_normalized=self.Ynorm
        )

        self.Xnorm_gradients = get_feature_gradients(self.mlp, self.Xnorm)

        self.max_fall, self.extremums, self.min_val_idx, self.max_val_idx = find_max_negative_slope(
            self.Xnorm_gradients,
            Y_original=self.Y
        )
    
    def _process(self, ohlcv_data: dict) -> None:
        self(ohlcv_data)  # get all the attributes
        return self.max_fall, self.min_val_idx, self.max_val_idx  # select ones you need

    def parallel_process(self, ohlcv_data_list: list, num_workers: int = 2) -> list:
        with mp.Pool(processes=num_workers) as pool:
            processed_data = pool.map(self._process, ohlcv_data_list)
        return processed_data
    
    def visualize_fall(self) -> None:
        if [self.max_fall, self.extremums, self.min_val_idx, self.max_val_idx] == [None, None, None, None]:
            print(f"{RED_BOLD}(guru){RESET} MaxFallFinder().visualize_fall()\n"
                  " ----> Unable to visualize fall on increasing graph.")
        else:
            visualize_model(self.mlp, self.X, self.Y, self.Xnorm, self.Ynorm, 
                            self.extremums, self.min_val_idx, self.max_val_idx)



# outdated
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
    grads = get_feature_gradients(model, Xn)
    slope, extremums, min_val_idx, max_val_idx = find_max_negative_slope(grads, Y)
    
    # if visualize_graph is True:
    #     visualize_model(model, X, Y, Xn, Yn, extremums, min_val_idx, max_val_idx)
    return slope.item()
