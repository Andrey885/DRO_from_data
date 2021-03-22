import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express, plotly.graph_objects, plotly.io


def main():
    exp = 'data_exp1'
    mean_hoef = np.load(f'{exp}/mean_hoef.npy')
    std_hoef = np.load(f'{exp}/std_hoef.npy')
    mean_dro = np.load(f'{exp}/mean_dro.npy')
    std_dro = np.load(f'{exp}/std_dro.npy')
    # mean_dro_cropped = np.load(f'{exp}/mean_dro_cropped.npy')
    # std_dro_cropped = np.load(f'{exp}/std_dro_cropped.npy')
    T_mins = np.load(f'{exp}/alphas.npy')

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
    fig = plotly.graph_objects.Figure([
        plotly.graph_objects.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name='Hoeffding'
        ),
        plotly.graph_objects.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
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
            x=x+x[::-1], # x, then x reversed
            y=y_upper_dro+y_lower_dro[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(100,0,80,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        #     plotly.graph_objects.Scatter(
        #         x=x,
        #         y=y_dro_cropped,
        #         line=dict(color='rgb(100,100,0)'),
        #         mode='lines',
        #         name='truncated DRO'
        #     ),
        #     plotly.graph_objects.Scatter(
        #         x=x+x[::-1], # x, then x reversed
        #         y=y_upper_dro_cropped+y_lower_dro_cropped[::-1], # upper, then lower reversed
        #         fill='toself',
        #         fillcolor='rgba(100,100,0,0.1)',
        #         line=dict(color='rgba(255,255,255,0)'),
        #         hoverinfo="skip",
        #         showlegend=False
        #     )
        ])
        # fig.update_layout(title=r"Hoeffding vs DRO vs truncated DRO, binomial C with uniform T, T_min=50",
        fig.update_layout(title=r"Hoeffding vs DRO, binomial, T_min=10",
                         xaxis_title="T_min",
                         yaxis_title="Expected loss")
        fig.show()
        # plotly.io.write_image(fig, exp + "_smoothed.jpg", width=1280, height=640)
        plotly.io.write_image(fig, exp + ".jpg", width=1280, height=640)


if __name__ == '__main__':
    main()
