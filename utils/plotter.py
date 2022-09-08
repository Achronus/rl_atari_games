from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: tuple[list, list], plot_type: str, format_dict: dict, figsize: tuple = (8, 6),
              graph_params: dict = None, filename: str = None) -> None:
    """
    Visualises the x and y data onto the chosen plot type.

    :param data (tuple[list, list]) - lists of data to display on the x-axis and y-axis (x, y)
    :param plot_type (string) - type of plot to create.
                                Available plots: ['line', 'scatter', 'bar', 'histogram', 'boxplot']
    :param format_dict (dict) - a dictionary containing plot formatting information.
                                Required keys: ['title', 'xlabel', 'ylabel']
                                Optional keys: ['disable_y_ticks', 'disable_x_ticks']
    :param figsize (tuple) - (optional) size of the plotted figure.
    :param graph_params (dict) - (optional) additional parameters unique to the selected graph. Refer to matplotlib
                                 documentation for more details.
    :param filename (string) - (optional) a filepath and name for saving the plot. E.g., '/ppo/cur/ppo-cur_SpaInv.png'.
    """
    valid_format_keys = ['title', 'xlabel', 'ylabel', 'disable_y_ticks', 'disable_x_ticks']
    if not isinstance(data, tuple):
        raise ValueError("'data' must be a tuple of '(x,)' or ('x, y)' data!")

    if len(data) > 2 or len(data) == 0:
        raise ValueError("Plottable data  must be: '(x,)' or '(x, y)'!")
    elif len(data) == 1 and plot_type in ['line', 'scatter', 'bar']:
        raise ValueError("Missing plottable 'y' data! Must be: 'data=(x, y)'!")
    elif len(data) == 2 and plot_type in ['histogram', 'boxplot']:
        raise ValueError("Too many plottable data points! Must be 'data=(x,)'!")

    for key in format_dict.keys():
        if key not in valid_format_keys:
            raise KeyError(f"Invalid key '{key}'! Required keys: ['title', 'xlabel', 'ylabel'] "
                           f"Optional keys: ['disable_y_ticks', 'disable_x_ticks']")

    fig, ax = plt.subplots(figsize=figsize)
    plotter = Plotter(ax, data, format_dict, graph_params)
    getattr(plotter, plot_type)()  # Create plot

    plt.xticks(rotation=90)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


class Plotter:
    """A basic class for visualising data that uses matplotlib graphs."""
    def __init__(self, ax: Any, data: tuple, format_dict: dict, graph_params: dict) -> None:
        self.ax = ax
        self.x = np.asarray(data[0])
        self.format_dict = format_dict
        self.graph_params = {} if graph_params is None else graph_params

        if len(data) == 2:
            self.y = np.asarray(data[1])

        self.ax.set_title(format_dict['title'])
        self.ax.set_xlabel(format_dict['xlabel'])
        self.ax.set_ylabel(format_dict['ylabel'])

        if 'disable_y_ticks' in format_dict.keys():
            self.ax.yaxis.set_ticks([])

        if 'disable_x_ticks' in format_dict.keys():
            self.ax.xaxis.set_ticks([])

    def line(self) -> None:
        """Visualises x and y data on a line chart."""
        self.ax.plot(self.x, self.y, **self.graph_params)

    def scatter(self) -> None:
        """Visualises x and y data on a scatter plot."""
        self.ax.scatter(self.x, self.y, **self.graph_params)

    def bar(self) -> None:
        """Visualises x and y data on a bar chart."""
        self.ax.bar(self.x, self.y, **self.graph_params)

    def histogram(self) -> None:
        """Visualises x data on a histogram."""
        self.ax.hist(self.x, **self.graph_params)

    def boxplot(self) -> None:
        """Visualises x data on a boxplot."""
        self.ax.boxplot(self.x, **self.graph_params)
