import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from argparse import Namespace
import plotly
import plotly.express, plotly.graph_objects, plotly.io


def main(exp, x_name, title, args):
    c_worst_string = '_c_worst' if args.costs == 'true' else ''
    mean_hoef = np.load(f'{exp}/mean{c_worst_string}_hoef.npy')
    std_hoef = np.load(f'{exp}/std{c_worst_string}_hoef.npy')
    mean_dro = np.load(f'{exp}/mean{c_worst_string}_dro.npy')
    std_dro = np.load(f'{exp}/std{c_worst_string}_dro.npy')
    T_mins = np.load(f'{exp}/params.npy')
    if os.path.exists(f'{exp}/mean{c_worst_string}_dro_cropped.npy'):
        mean_dro_cropped = np.load(f'{exp}/mean{c_worst_string}_dro_cropped.npy')
        std_dro_cropped = np.load(f'{exp}/std{c_worst_string}_dro_cropped.npy')
    else:
        mean_dro_cropped = std_dro_cropped = np.zeros(len(T_mins))

    y_axis = "Expected loss"
    if args.costs == 'true':
        T_mins = np.linspace(0, mean_hoef.shape[1] - 1, mean_hoef.shape[1]).astype(int)
        mean_hoef = mean_hoef[0]
        std_hoef = std_hoef[0]
        mean_dro = mean_dro[0]
        std_dro = std_dro[0]
        mean_dro_cropped = mean_dro_cropped[0]
        std_dro_cropped = std_dro_cropped[0]
        y_axis = "Costs estimation"
        x_name = "Cost number (sorted by nominal value)"
    if args.percentage_mode == 'true':
        y_axis = "Outperforming rate"
    x = T_mins.tolist()
    y = mean_hoef.tolist()
    y_upper = (mean_hoef + std_hoef).tolist()
    y_lower = (mean_hoef - std_hoef).tolist()
    y_dro = mean_dro.tolist()
    y_upper_dro = (mean_dro + std_dro).tolist()
    y_lower_dro = (mean_dro - std_dro).tolist()
    y_dro_cropped = mean_dro_cropped.tolist()
    y_upper_dro_cropped = (mean_dro_cropped + std_dro_cropped).tolist()
    y_lower_dro_cropped = (mean_dro_cropped - std_dro_cropped).tolist()
    graphs = [
        plotly.graph_objects.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name='Hoeffding'
        ),
        plotly.graph_objects.Scatter(
            x=x+x[::-1],  # x, then x reversed
            y=y_upper+y_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        plotly.graph_objects.Scatter(
            x=x,
            y=y_dro,
            line=dict(color='rgb(100,0,80)'),
            mode='lines',
            name='DRO'
        ),
        plotly.graph_objects.Scatter(
            x=x+x[::-1],  # x, then x reversed
            y=y_upper_dro+y_lower_dro[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(100,0,80,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )]
    if np.std(y_dro_cropped) != 0:
        if args.count_cropped == 'true':
            third_axis_title = 'truncated DRO'
        elif args.percentage_mode == 'true':
            third_axis_title = 'equal rate'
        elif args.costs == 'true':
            third_axis_title = 'nominal costs'
        graphs.extend([plotly.graph_objects.Scatter(
            x=x,
            y=y_dro_cropped,
            line=dict(color='rgb(100,100,0)'),
            mode='lines',
            name=third_axis_title
        ),
        plotly.graph_objects.Scatter(
            x=x+x[::-1],  # x, then x reversed
            y=y_upper_dro_cropped+y_lower_dro_cropped[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(100,100,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )])
    fig = plotly.graph_objects.Figure(graphs)
    fig.update_layout(title=title,
                      xaxis_title=x_name,
                      yaxis_title=y_axis)
    # y_min = 0.95
    # y_max = np.quantile(y_upper_dro, 0.9)
    # fig.update_yaxes(range=[y_min, y_max])
    # fig.show()
    plotly.io.write_image(fig, f"{exp}/graph.jpg", width=1280, height=640)


if __name__ == '__main__':
    exp = 'exp15'
    title = "Hoeffding vs DRO, binomial, T_min=10"
    x_name = "T_min"
    with open(f'{exp}/args.json', 'r') as f:
        args = json.load(f)
    args = Namespace(**args)
    main(exp, title, x_name, args)
