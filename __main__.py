from os.path import join
from sklearn.externals import joblib

from grasping_position_inference.training.model_generator import generate_models
from grasping_position_inference.inference.model import Model

MODEL_PATH = 'models'


def get_probability_distribution_for_grid(x, y, model_name):
    model_filepath = join(MODEL_PATH, model_name)
    model = joblib.load(model_filepath)

    result = []

    for i in range(0, len(x)):
        success_rate = []
        for predict_values in model.predict_proba(map(list, zip(x[i], y[i]))):
            success_rate.append(predict_values[1])
        result.append(success_rate)

    return result


if __name__ == "__main__":
    #generate_models()
    model = Model('cup.n.01', 'BACK', 'BOTTOM', 'pr2_left_arm')
    model.load()
