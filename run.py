import json
import approguru as guru


# Release Notes:
#   - Nothing about user interface has changed (yay!)

#   - In case Guru receives the graph decreasing in the beginning it won't crop it as before but adjust the search area to consider
#   the whole trend. Therefore, more computation is done underhood allowing guru freely moving the starting pointer within some range.
#   This range is controlled by (FETCHED_DATA_POINTS - INITIAL_SEARCH_REGION) difference you can play with in config.
#   Important note! Guru aborts execution after FETCHED_DATA_POINTS == len(your_input_ohlcv_list) condition is not met.

#   - Buffer search is now applied to every negative trend Guru stumbles upon before It picks the most one. Thus, the risk of skipping
#   correct answer after bulding a valid approximation is gone.

#   - After finding fall boundary Guru picks most and least values among open/close values on the left and right accordingly, taking
#   one more minor step to better accuracy.

# same ohlcv as always
with open("data/HOUR_data.json", "r") as file:
    data = json.load(file)


# create object, then simply pass ohlcv to it
finder = guru.MaxFallFinder(seed=245346)  # seed is purely optional, Guru stiks to 13 by default
finder(data)


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