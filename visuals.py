import sys
import torch
import plotly.graph_objects as go
from datetime import datetime
from utilities import MLP, polynomial_features
from globals import *


def timestamps_to_dates(timestamps: torch.tensor):
    """
    Convert a tensor of timestamps to a tensor of date strings.
    """
    format = "%d-%m-%Y %H:%M:%S"
    date_strings = [datetime.fromtimestamp(ts.item()).strftime(format) for ts in timestamps]
    return date_strings


def visualize_model(model: MLP, X_normalized: torch.tensor, Y_normalized: torch.tensor) -> None:
        # X: torch.tensor, Y: torch.tensor, predictions: torch.tensor,
        #             extremums: torch.tensor, fall_start: int, fall_end: int) -> None:
    """
    
    """
    # assert model.visualized_data is not None, "No data was obtained from find_max_negative_slope()"
    Xn, Yn = X_normalized, Y_normalized

    try:
        model.visualized_data
    except AttributeError as e:
        print("'MLP' object has no attribute 'visualized_data'")
        # sys.exit(1)
    X, Y, extremums, mini, maxi = model.visualized_data  # Unpack data from find_max_negative_slope() 
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
        x0=Xnplot[mini.item()],
        x1=Xnplot[maxi.item()],
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
        xaxis=dict(
            showline=False,
            zeroline=False, 
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="rgba(211, 211, 211, 0.4)",
            gridwidth=0.05, 
            showline=False,
            zeroline=False,
            showticklabels=False
        )
    )
    fig.show()