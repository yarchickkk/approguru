import json
from utilities import magic_function

with open("data/token5_data.json", "r") as file:
    data5 = json.load(file)

result = magic_function(data5)
print(f"Slope: {result:+.10f}")
