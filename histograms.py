import matplotlib.pyplot as plt
import numpy as np


# Author : Baris
class SlidingWindow():
    WINDOW_LENGTH = int(40)
    X_TICK_STEP = int(5)

    def __init__(self, name, stats_names, width, height, dpi=96):
        self.name = name
        self.stats_names = stats_names
        plt.ion()
        # Figure and axis
        self.fig = None
        self.ax = None
        # Dimensions
        self.dpi = dpi
        self.width = float(width)/self.dpi
        self.height = float(height)/self.dpi
        # Decorations
        self.title = None
        self.x_label = None
        self.y_label = None
        self.x_tick_labels = np.arange(self.X_TICK_STEP - self.WINDOW_LENGTH, 1, self.X_TICK_STEP)
        self.x_tick_pos = np.arange(self.X_TICK_STEP-1, self.WINDOW_LENGTH, self.X_TICK_STEP) + 0.5
        self.y_tick_labels = None
        self.y_tick_pos = None
        self.legend = None
        self.legend_loc = 0
        # Time
        self.epoch = 0

    def create(self):
        self.fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi, frameon=False)
        self.ax = self.fig.gca()

    def slide_x_ticks(self):
        if self.epoch % self.X_TICK_STEP == 0:
            self.x_tick_pos += self.X_TICK_STEP - 1
            self.x_tick_labels += self.X_TICK_STEP
        else:
            self.x_tick_pos -= 1

    def draw(self, slide_x_ticks=True):
        # update time
        self.epoch += 1
        # x ticks
        if slide_x_ticks:
            self.slide_x_ticks()
        self.ax.set_xticks(self.x_tick_pos, minor=False)
        self.ax.set_xticklabels(self.x_tick_labels)
        # y-ticks
        if self.y_tick_labels is not None:
            if self.y_tick_pos is not None:
                self.ax.set_yticks(self.y_tick_pos, minor=False)
            self.ax.set_yticklabels(self.y_tick_labels)
        # labels and titles
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        if self.legend:
            self.ax.legend(self.legend, loc=self.legend_loc)
        # draw
        self.fig.canvas.draw()
        plt.pause(0.1)
