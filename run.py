import json
import approguru as guru
from get_data import pools
import torch


# Load data in a list
ohlcv_data = []
for idx, pool in enumerate(pools):
    with open(f"data/{str(idx)}_data.json", "r") as file:
        data = json.load(file)
    ohlcv_data.append(data)


# Process data one-by-one
for idx, data in enumerate(ohlcv_data):
    break
    if idx != 10:
        continue

    finder = guru.MaxFallFinder()
    finder(data)
    # logging
    print(f"Graph {idx}:")
    print(f"Max fall: {finder.max_fall.item()}, length: {(finder.min_val_idx - finder.max_val_idx).item()}, ({finder.max_val_idx.item()}, {finder.min_val_idx.item()})")
    print(f"Loss: {finder.achieved_loss.item():.10f} | Iters: {finder.steps_made.item()}")
    print("-"*30)
    finder.visualize_fall()


with open("data/test_data.json", "r") as file:
    data = json.load(file)
finder = guru.MaxFallFinder()
finder(data)
print(f"Max fall: {finder.max_fall.item()}, length: {(finder.min_val_idx - finder.max_val_idx).item()}, ({finder.max_val_idx.item()}, {finder.min_val_idx.item()})")
print(f"Loss: {finder.achieved_loss.item():.10f} | Iters: {finder.steps_made.item()}")
finder.visualize_fall()

    # b, _ = torch.min(finder.data[finder.min_val_idx][1, 2, 3, 4])
    # print(f"Values:\nMin: {b.item()}]nMax: {a.item()}")

# MaxFallFinder() main attributes:
#   - max_fall (torch.Tensor)     : most fall detected (in %)
#   - min_val_idx (torch.Tensor)  : candle closing the fall (greater index)
#   - max_val_idex (torch.Tensor) : candle starting the fall (smaller index)
#   - steps_made (torch.Tensor)   : optimization duration
#   - achieved_loss (torch.Tensor): how well model fits the graph
#   - start_ptr (torch.Tensor)    : index of a candle search area begins with

# To pop out numerical value out of the attribute just use .item()
#   - finder.max_fall       : torch.Tensor([0.2739475)
#   - finder.max_fall.item(): 0.2739475 (python float)



# # print(f"fall begins: {finder.max_val_idx.item()}, price: {finder.Y[finder.max_val_idx].item()}")
# # print(f"fall ends  : {finder.min_val_idx.item()}, price: {finder.Y[finder.min_val_idx].item()}")
# # print(f"fall: {((finder.Y[finder.max_val_idx] / finder.Y[finder.min_val_idx] - 1.0) * 100).item()} %")
# # print(finder.extremums)
# # 1731533108

"""Xnorm = torch.linspace(0, 1, 1000).view(-1, 1)
Xpoly = polynomial_features(Xnorm, INPUT_SIZE)

model = finder.mlp
Ynorm = model(Xpoly)

Xnplot = Xnorm.detach().view(-1).numpy()
Ynplot = Ynorm.detach().view(-1).numpy()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Xnplot, 
    y=Ynplot,
    line=dict(color='orange')
))

fig.add_trace(go.Scatter(
    x=finder.Xnorm.detach().view(-1).numpy(),
    y=finder.Ynorm.detach().view(-1).numpy(),
    line=dict(color='blue')
))
fig.show()"""
