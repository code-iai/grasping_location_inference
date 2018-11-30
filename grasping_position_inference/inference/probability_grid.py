import numpy as np

STEP_SIZE = 0.01
MIN_X = -0.8
MIN_Y = -0.8
MAX_X = 0.8
MAX_Y = 0.8

INIT_PROBABILITY = 0.5


class ProbabilityGrid(object):
    def __init__(self):
        self._grid = _init_grid()

    def __getitem__(self, key):
        x, y = _transform_key_to_grid_coordinates(key)
        return self._grid[x][y]

    def __setitem__(self, key, value):
        x, y = _transform_key_to_grid_coordinates(key)
        self._grid[x][y] = value

    def update(self, predictor):
        inference_result = predictor.get_probability_distribution_for_grid()

        min_x, min_y, max_x, max_y = predictor._min_x, predictor._min_y, predictor._max_x, predictor._max_y
        x_steps = _steps(min_x, max_x)
        y_steps = _steps(min_y, max_y)

        for x in range(0, x_steps):
            current_x = min_x + (x*STEP_SIZE)
            for y in range(0, y_steps):
                current_y = min_y + (y*STEP_SIZE)
                self[current_x, current_y] = inference_result[x][y]



def _init_grid():
    dimension = _steps(-0.8, 0.8)
    dimension = dimension + 1

    return np.full((dimension, dimension), INIT_PROBABILITY)


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

def _transform_key_to_grid_coordinates(key):
    x = _steps(MIN_X, key[0])
    y = _steps(key[1], MAX_Y)

    return x, y