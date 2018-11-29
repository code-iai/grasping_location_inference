import numpy as np


class ProbabilityGrid(object):
    def __init__(self):
        self._grid = _init_grid()

    def __getitem__(self, key):
        return self._grid[key[0], key[1]]


def _init_grid():
    dimension = 1.6 / 0.01
    dimension = int(round(dimension, 0)) + 1

    return np.full((dimension, dimension), 0.5)