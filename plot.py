import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from argparse import Namespace
import plotly
import plotly.express, plotly.graph_objects, plotly.io


def main(exp, x_name, title, args, count_percentage=False, count_costs=False):
    solutions_hoef = np.load(f'{exp}/solutions_hoef.npy')
    solutions_dro = np.load(f'{exp}/solutions_dro.npy')
    solutions_dro_cropped = np.load(f'{exp}/solutions_dro_cropped.npy')
    c_bar = np.load(f'{exp}/c_bar.npy')
    c_worst_dro = np.load(f'{exp}/c_worst_dro.npy')
    c_worst_hoef = np.load(f'{exp}/c_worst_hoef.npy')
    params = np.load(f'{exp}/params.npy')
    if count_percentage:
        std_dro = std_dro_cropped = std_hoef = np.zeros(len(params))
        solutions_hoef_perc = np.zeros_like(solutions_hoef)
        solutions_dro_perc = np.zeros_like(solutions_dro)
        solutions_eq_perc = np.zeros_like(solutions_dro)
        solutions_eq_perc[solutions_hoef == solutions_dro] = 1
        solutions_dro_perc[solutions_hoef > solutions_dro] = 1
        solutions_hoef_perc[solutions_hoef < solutions_dro] = 1
        solutions_hoef = solutions_hoef_perc
        solutions_dro = solutions_dro_perc
        solutions_dro_cropped = solutions_eq_perc
    else:
        std_c_worst_dro = np.median(np.abs(c_worst_dro - np.mean(c_worst_dro)), axis=0)
        std_c_worst_hoef = np.median(np.abs(c_worst_hoef - np.mean(c_worst_hoef)), axis=0)
        std_c_bar = np.median(np.abs(c_bar - np.mean(c_bar)), axis=0)
        std_dro = np.median(np.abs(solutions_dro - np.mean(solutions_dro)), axis=0)
        std_dro_cropped = np.median(np.abs(solutions_dro_cropped - np.mean(solutions_dro_cropped)), axis=0)
        std_hoef = np.median(np.abs(solutions_hoef - np.mean(solutions_hoef)), axis=0)
    if count_costs:
        solutions_hoef = c_worst_hoef[:, -3, :]
        solutions_dro = c_worst_dro[:, -3, :]
        solutions_dro_cropped = c_bar[:, -3, :]
        # std_hoef = np.median(np.abs(solutions_hoef - np.median(solutions_hoef, axis=0)), axis=0)
        # std_dro = np.median(np.abs(solutions_dro - np.median(solutions_dro, axis=0)), axis=0)
        # std_dro_cropped = np.median(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped, axis=0)), axis=0)
        # std_dro_cropped = np.zeros(len(std_hoef))
        std_dro = std_dro_cropped = std_hoef = np.zeros(solutions_dro_cropped.shape[1])
    mean_hoef = np.mean(solutions_hoef, axis=0)
    mean_dro = np.mean(solutions_dro, axis=0)
    mean_dro_cropped = np.mean(solutions_dro_cropped, axis=0)

    mean_c_worst_dro = np.mean(c_worst_dro, axis=0)
    mean_c_worst_hoef = np.mean(c_worst_hoef, axis=0)
    mean_c_bar = np.mean(c_bar, axis=0)
    y_axis = "Nominal relative loss"
    if count_costs:
        params = np.linspace(0, c_worst_dro.shape[-1] - 1, c_worst_dro.shape[-1]).astype(int)
        y_axis = "Costs estimation"
        x_name = "arc index (in the sorted array)"
    if count_percentage == 'true':
        y_axis = "Outperforming rate"

    if args.count_cropped2 == 'DRO':
        data_passed_as_hoeffding_name = 'truncated DRO 2'
        data_passed_as_dro_name = 'baseline DRO'
    elif args.count_cropped2 == 'Hoeffding':
        data_passed_as_hoeffding_name = 'Hoeffding bounds'
        data_passed_as_dro_name = 'Hoeffding bounds truncated'
    else:
        data_passed_as_hoeffding_name = 'Hoeffding bounds'
        data_passed_as_dro_name = 'baseline DRO'
    x = params.tolist()
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
            line=dict(color='rgb(250,0,0)'),
            mode=args.plot_mode,
            name=data_passed_as_hoeffding_name
        ),
        plotly.graph_objects.Scatter(
            x=x,
            y=y_dro,
            line=dict(color='rgb(0,250,0)'),
            mode=args.plot_mode,
            name=data_passed_as_dro_name
        )
        ]
    if np.max(std_dro) != 0:
        graphs.extend([plotly.graph_objects.Scatter(
            x=x+x[::-1],  # x, then x reversed
            y=y_upper+y_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(100,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        plotly.graph_objects.Scatter(
            x=x+x[::-1],  # x, then x reversed
            y=y_upper_dro+y_lower_dro[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )])
    if np.std(y_dro_cropped) != 0:
        if count_percentage:
            third_axis_title = 'equal rate'
        elif count_costs:
            third_axis_title = 'nominal costs'
        elif args.count_cropped == 'true':
            third_axis_title = 'truncated DRO'
        graphs.extend([plotly.graph_objects.Scatter(
            x=x,
            y=y_dro_cropped,
            line=dict(color='rgb(0,0,250)'),
            mode=args.plot_mode,
            name=third_axis_title
        )])
        if np.max(std_dro) != 0:
            graphs.append(plotly.graph_objects.Scatter(
                x=x+x[::-1],  # x, then x reversed
                y=y_upper_dro_cropped+y_lower_dro_cropped[::-1],  # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,0,100,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
    fig = plotly.graph_objects.Figure(graphs)
    fig.update_layout(title='',
                      xaxis_title=x_name,
                      yaxis_title=y_axis)
    plotly.io.write_image(fig, f"{exp}/graph_{title}.jpg", width=1280, height=640)


if __name__ == '__main__':
    exp = 'exp8b'
    with open(f'{exp}/args.json', 'r') as f:
        args = json.load(f)
    args = Namespace(**args)
    args.costs = 'false'
    title = "loss" if args.costs == 'false' else 'costs'
    x_name = "delta"
    main(exp, x_name, title, args, count_costs = args.costs == 'true')
