import json
import torch

with open("data/token5_data.json", "r") as file:
    data5 = json.load(file)

import approguru as guru
# result = guru.core.magic_function(data5)
# if result is not None:
#     print(f"Slope: {result:+.10sf}")

# import approguru as guru
print(f"Using defice: {guru.device}")
finder = guru.MaxFallFinder()

finder(data5)
print(f"Max Slope: {finder.max_fall}")
