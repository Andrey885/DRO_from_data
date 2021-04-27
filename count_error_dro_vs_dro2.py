import numpy as np
import networkx
import plotly
import os
import json

import main
import graph_utils


def run():
    exp = 'exp8'
    os.makedirs(exp, exist_ok=True)
    args = main.parse_args()
    args.normal_std = 50/4
    args.mode = 'normal'
    args.w = args.h = 3
    args.delta = 20
    args.T_min = 10
    args.count_cropped = 'true'
    args.count_cropped2 = 'DRO'
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]
    all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]
    solutions_dro2 = solutions_dro = 0
    while solutions_dro2 >= solutions_dro:
        solutions_dro2, solutions_dro, solutions_dro_cropped, c_worst_dro, c_worst_dro2, c_bar, _, _ = main.run_graph(g, edges_num_dict,
                                                                                                            args, start_node,
                                                                                                            finish_node, all_paths=all_paths)
    params = np.linspace(0, c_worst_dro.shape[-1] - 1, c_worst_dro.shape[-1]).astype(int)
    y_axis = "Expected costs"
    x_name = "arc index (in the sorted array)"
    x = params.tolist()
    graphs = [
        plotly.graph_objects.Scatter(
            x=x,
            y=c_worst_dro2.tolist(),
            line=dict(color='rgb(0,87,31)', dash='dash'),
            mode=args.plot_mode,
            name='truncated DRO 2'
        ),
        plotly.graph_objects.Scatter(
            x=x,
            y=c_worst_dro.tolist(),
            line=dict(color='rgb(0,87,31)'),
            mode=args.plot_mode,
            name='baseline DRO'
        ),
        plotly.graph_objects.Scatter(
            x=x,
            y=c_bar.tolist(),
            line=dict(color='rgb(0,0,0)'),
            mode=args.plot_mode,
            name='nominal costs'
        )]
    fig = plotly.graph_objects.Figure(graphs)
    fig.update_layout(title='',
                      xaxis_title=x_name,
                      yaxis_title=y_axis,
                      plot_bgcolor='rgba(200,200,200,0)')

    plotly.io.write_image(fig, f"{exp}/graph.jpg", width=1280, height=640)

    num_edges = c_bar.shape[0]
    errors_dro2 = errors_dro = 0
    for i in range(num_edges):
        for j in range(num_edges):
            if c_bar[i] < c_bar[j] and c_worst_dro2[i] > c_worst_dro2[j]:
                errors_dro2 += 1
            if c_bar[i] < c_bar[j] and c_worst_dro[i] > c_worst_dro[j]:
                errors_dro += 1

    print("Number of errors DRO:", errors_dro)
    print("Number of errors DRO2:", errors_dro2)
    with open(exp + "/errors_count.json", 'w') as f:
        json.dump({"errors_num_DRO": errors_dro, "errors_num_DRO_2": errors_dro2}, f)

if __name__ == '__main__':
    run()
