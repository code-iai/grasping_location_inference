# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility

import re
from os import listdir, path
from grasping_position_inference.root import ABSOLUTE_PATH

MODEL_PATH = path.join(ABSOLUTE_PATH, 'models')


class Model(object):
    def __init__(self, *evidences):
        self.grasping_object_type, self.grasping_type, self.bottom_face, self.arm = evidences
        self._regex = self._prepare_regex_for_model_filtering()
        self.predictors = []

    def _prepare_regex_for_model_filtering(self):
        return re.escape("{},{},{},{}".format(self.grasping_object_type, self.grasping_type, self.bottom_face, self.arm)) + r".+$"

    def load(self):
        models = '\n'.join(listdir(MODEL_PATH))
        matches = re.finditer(self._regex, models, re.MULTILINE)
        for match in matches:
            file_name = match.group()