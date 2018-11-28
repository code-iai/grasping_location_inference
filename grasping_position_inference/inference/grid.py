import numpy as np


class Grid(object):
    def __init__(self, x_definition, y_definition):
        self._x_start, self._x_end, self._x_step_size = x_definition
        self._y_start, self._y_end, self._y_step_size = y_definition

        #It is required to add the step size to create the real required grid size
        #Numpy revokes the last cordinate during the grid creation

        self._x_end += self._x_step_size
        self._y_end += self._y_step_size

        self.grid = self._create_grid()

    def _create_grid(self):
        x = np.arange(self._x_start, self._x_end, self._x_step_size)
        y = np.arange(self._y_start, self._y_end, self._y_step_size)

        return np.meshgrid(x, y)