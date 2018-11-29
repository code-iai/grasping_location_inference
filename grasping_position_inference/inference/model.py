# coding=utf8

import re
from os import listdir, path
import numpy as np

from grasping_position_inference.inference.predicator import Predicator
from grasping_position_inference.root import ABSOLUTE_PATH
from grasping_position_inference.inference.grid import Grid


MODEL_PATH = path.join(ABSOLUTE_PATH, 'models')


class Model(object):
    def __init__(self, *evidences):
        self.grasping_object_type, self.grasping_type, self.bottom_face, self.arm = evidences
        self._regex = self._prepare_regex_for_model_filtering()
        self.predictors = []

    def _prepare_regex_for_model_filtering(self):
        return re.escape("{},{},{},{}".format(
            self.grasping_object_type, self.grasping_type, self.bottom_face, self.arm)) + r".+$"

    def load(self):
        models = '\n'.join(listdir(MODEL_PATH))
        matches = re.finditer(self._regex, models, re.MULTILINE)

        for match in matches:
            file_name = match.group()
            predicator = Predicator(file_name)
            self.predictors.append(predicator)

    def get_probability_distribution_for_grid(self):
        result_grid = _init_result_grid()
        result_grid[0][0] = 1.

        for predictor in self.predictors:
            inference_result = predictor.get_probability_distribution_for_grid()


def _init_result_grid():
    dimension = 1.6/0.01
    dimension = int(round(dimension, 0)) + 1

    return np.full((dimension, dimension), 0.5)

