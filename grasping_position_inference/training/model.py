from os.path import join
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from grasping_position_inference.root import ABSOLUTE_PATH

from grasping_position_inference.training.exceptions import DataSetIsEmpty, ModelIsNotTrained

MODEL_PATH = join(ABSOLUTE_PATH, 'models')
DATA_PATH = join(ABSOLUTE_PATH, 'data')


class Model(object):
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.grasping_object_type, self.grasping_type, self.robot_face, self.bottom_face, self.arm \
            = self._parse_data_filename()
        self._data = self._read_data()
        self._trained_model = None

    def _parse_data_filename(self):
        grasping_object_type, grasping_type, faces, arm = self.data_filename.split(',')
        arm = arm.split('.')[0]
        robot_face, bottom_face = faces.split()
        robot_face = robot_face.replace(':', '')
        bottom_face = bottom_face.replace(':', '')

        return grasping_object_type, grasping_type, robot_face, bottom_face, arm

    def _read_data(self):
        data_filepath = join(DATA_PATH, self.data_filename)
        return pd.read_csv(data_filepath, sep=',')

    def train(self):
        if self._data.empty:
            error_message = 'The empty data set "{}" cannot be used for training.'.format(self.data_filename)
            raise DataSetIsEmpty(error_message)

        features = self._data[['t_x', 't_y', 't_z']]
        labels = self._data["success"].map(lambda x: 1 if x else 0)

        gnb = GaussianNB()

        self._trained_model = gnb.fit(features, labels)

    def store(self):
        if self._trained_model is None:
            error_message = 'The model has to be trained before it can be stored.'
            raise ModelIsNotTrained(error_message)

        model_name = '{},{},{},{},{}.model'.format(self.grasping_object_type,
                                                   self.grasping_type,
                                                   self.robot_face,
                                                   self.bottom_face,
                                                   self.arm)
        model_save_path = join(MODEL_PATH, model_name)
        joblib.dump(self._trained_model, model_save_path)
