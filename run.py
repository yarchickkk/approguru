import json
import time
import torch
import plotly.graph_objects as go


with open("data/token1_data.json", "r") as file:
    data1 = json.load(file)

with open("data/token2_data.json", "r") as file:
    data2 = json.load(file)

with open("data/token3_data.json", "r") as file:
    data3 = json.load(file)

with open("data/token4_data.json", "r") as file:
    data4 = json.load(file)

with open("data/token5_data.json", "r") as file:
    data5 = json.load(file)



import approguru as guru
print(f"Using device: {guru.device}.")
print(f"CPU threads : {guru.num_threads}.")
print()
finder = guru.MaxFallFinder()
opt_finder = torch.compile(finder)


c, e = [], []
for data in [data1, data2, data3, data4, data5]:
    start = time.perf_counter()
    opt_finder(data)
    end = time.perf_counter()
    print("Seconds:", end - start)
    c.append(end - start)


for data in [data1, data2, data3, data4, data5]:
    start = time.perf_counter()
    finder(data)
    end = time.perf_counter()
    print("Seconds:", end - start)
    e.append(end - start)
    

# start = time.perf_counter()
# guru.magic_function(data1, visualize_graph=False)
# end = end = time.perf_counter()
# print("seconds:", end - start)
fig = go.Figure()
fig.add_trace(go.Scatter(  # approximated function
    x=[1, 2, 3, 4, 5], 
    y=e, 
    mode='lines+markers', 
    name='Eager', 
    ))

fig.add_trace(go.Scatter(  # approximated function
    x=[1, 2, 3, 4, 5], 
    y=c, 
    mode='lines+markers', 
    name='Compiled', 
    ))

fig.update_layout(  # display properties
    template='plotly',
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
    )
)

fig.show()