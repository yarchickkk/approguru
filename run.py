import json
import approguru as guru


# Release Notes:
#   - Added manual seed setting before every call of something random-related

#   - Added and now using get_feature_gradients_v2(), which sets feature gradients by 
#   looking at value of the next approximation point we're interested in. If it's greater
#   then the gradient is positive, in opposite scenario it's negative.
#   Previously guru looked at what torch was saying in this exact point and could sometimes
#   skip whole trends if he was unlucky enough.

# same ohlcv as always
with open("data/HOUR_data.json", "r") as file:
    data = json.load(file)


# create object, then simply pass ohlcv to it
finder = guru.MaxFallFinder()
finder(data)

print("RESULT:", round(float(finder.max_fall)) * 100)

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


# example of debugging:
print(f"Max fall: {finder.max_fall.item()}, length: {finder.min_val_idx - finder.max_val_idx}")
print(f"Loss: {finder.achieved_loss.item():.10f} | Iters: {finder.steps_made.item()}")

# opens up a nice picture in browser
finder.visualize_fall()

# print(f"fall begins: {finder.max_val_idx.item()}, price: {finder.Y[finder.max_val_idx].item()}")
# print(f"fall ends  : {finder.min_val_idx.item()}, price: {finder.Y[finder.min_val_idx].item()}")
# print(f"fall: {((finder.Y[finder.max_val_idx] / finder.Y[finder.min_val_idx] - 1.0) * 100).item()} %")
# print(finder.extremums)
# 1731533108

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
