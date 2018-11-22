import sys
from os import listdir
from os.path import isdir, join
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

MODEL_PATH = 'models'
DATA_PATH = 'data'


def train_model(data):
    features = data[['t_x', 't_y', 't_z']]
    labels = data["success"].map(lambda x: 1 if x else 0)

    gnb = GaussianNB()

    return gnb.fit(features, labels)


def generate_models():
    for data_filename in listdir(DATA_PATH):
        data_filepath = join(DATA_PATH, data_filename)
        grasping_object, grasping_type, faces, arm, _ = data_filename.split('.')
        robot_face, bottom_face = faces.split()

        data = pd.read_csv(data_filepath, sep=',')

        if not data.empty:
            model = train_model(data)

            model_name = '{}.{}.{}.{}.{}.model'.format(grasping_object, grasping_type, robot_face, bottom_face, arm)
            model_save_path = join(MODEL_PATH, model_name)
            joblib.dump(model, model_save_path)


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
    generate_models()