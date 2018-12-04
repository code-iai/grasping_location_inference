# coding=utf8

import re
from os import listdir, path

from grasping_position_inference.inference.predicator import Predicator
from grasping_position_inference.root import ABSOLUTE_PATH
from grasping_position_inference.inference.probability_grid import ProbabilityGrid


MODEL_PATH = path.join(ABSOLUTE_PATH, 'models')


class Model(object):
    def __init__(self):
        self.predictors = []

    def add_predictor(self, *evidences):
        grasping_object_type, grasping_type, robot_face, bottom_face, arm = evidences
        predicator_name = "{},{},{},{},{},".format(grasping_object_type, grasping_type,bottom_face, arm, robot_face)
        models = listdir(MODEL_PATH)
        file_name = ''

        for model in models:
            if model.startswith(predicator_name):
                file_name = model
                break

        if file_name:
            predicator = Predicator(file_name)
            self.predictors.append(predicator)

    def get_probability_distribution_for_grid(self):
        probability_grid = ProbabilityGrid()

        for predictor in self.predictors:
            probability_grid.update(predictor)

        return probability_grid.get_grid()





