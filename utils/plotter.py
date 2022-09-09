from typing import Any
import numpy as np


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
