from os import listdir, path

from grasping_position_inference.training.exceptions import DataSetIsEmpty
from grasping_position_inference.training.model import Model
from grasping_position_inference.root import ABSOLUTE_PATH

DATA_PATH = path.join(ABSOLUTE_PATH, 'data')


def generate_models(data_path=DATA_PATH):
    for data_filename in listdir(data_path):
        if data_filename.endswith('.csv'):
            model = Model(data_filename)
            try:
                model.train()
                model.store()
            except DataSetIsEmpty:
                print 'Skipping {} since it is empty'.format(data_filename)

