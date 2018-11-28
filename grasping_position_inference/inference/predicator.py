from os.path import join
from grasping_position_inference.root import ABSOLUTE_PATH
from sklearn.externals import joblib

MODEL_PATH = join(ABSOLUTE_PATH, 'models')


class Predicator(object):
    def __init__(self, file_name):
        self._file_name = file_name
        self._model_filepath = join(MODEL_PATH, file_name)
        self._min_x, self._max_x, self._min_y, self._max_y = self._get_min_max_values_for_x_y()

    def _get_min_max_values_for_x_y(self):
        #cup.n.01,BACK,BOTTOM,pr2_left_arm,BACK,-0.8;-0.5,0.0;0.5,.model
        splitted_file_name = self._file_name.split(',')
        x_range, y_range = splitted_file_name[-3],splitted_file_name[-2]
        min_x, max_x = map(float, x_range.split(';'))
        min_y, max_y = map(float, y_range.split(';'))

        return min_x, max_x, min_y, max_y



    def get_probability_distribution_for_grid(self, x, y, model_name):
        model = joblib.load(self._model_filepath)

        result = []

        for i in range(0, len(x)):
            success_rate = []
            for predict_values in model.predict_proba(map(list, zip(x[i], y[i]))):
                success_rate.append(predict_values[1])
            result.append(success_rate)

        return result