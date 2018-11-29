import numpy as np

STEP_SIZE = 0.01
MIN_X = -0.8
MIN_Y = -0.8
MAX_X = 0.8
MAX_Y = 0.8


class ProbabilityGrid(object):
    def __init__(self):
        self._grid = _init_grid()

    def __getitem__(self, key):
        return self._grid[key[0]][key[1]]


def _init_grid():
    dimension = 1.6 / STEP_SIZE
    dimension = int(round(dimension, 0)) + 1

    return np.full((dimension, dimension), 0.5)


def _steps(start, end):
    biggest = abs(end)
    smallest = abs(start)

    if smallest > biggest:
        temp = smallest
        smallest = biggest
        biggest = temp

    if np.sign(start) == np.sign(end):
        distance = biggest - smallest
    else:
        distance = biggest + smallest

    return int(round(distance / STEP_SIZE, 0))
