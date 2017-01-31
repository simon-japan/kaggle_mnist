import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps
from os import path
from sklearn.model_selection import cross_val_score


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        value = func(*args, **kwargs)
        print("Execution time:", (datetime.now() - t0))
        return value
    return wrapper


def _get_data(file_path):
    print("Reading in data from {}".format(file_path))
    df = pd.read_csv(file_path)
    print("Done reading.")
    return df.as_matrix()


def get_train_data(file_path, limit=None):
    data = _get_data(file_path)
    np.random.shuffle(data)
    x = data[:, 1:] / 255.0  # data is from 0..255
    y = data[:, 0]
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


def get_test_data(file_path):
    return _get_data(file_path)


def write_predictions(predictions, file_path):
    df = pd.DataFrame({
        "ImageId": list(range(1, len(predictions)+1)),
        "Label": predictions
    })
    df.to_csv(file_path, index=False, header=True)


def show_figure(x):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()


class ModelTester(object):
    def __init__(self, wd):
        self.working_directory = wd
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.model = None
        self.model_name = None
        self.predictions = None
        self.full_data_loaded = False

    @timed
    def load_train_data(self, limit=None):
        self.train_x, self.train_y = get_train_data(file_path=path.join(self.working_directory, 'train.csv'),
                                                    limit=limit)
        if not limit:
            self.full_data_loaded = True

    @timed
    def load_test_data(self):
        self.test_x = get_test_data(path.join(self.working_directory, 'test.csv'))

    def set_model(self, m, name):
        self.model = m
        self.model_name = name

    @timed
    def cross_validate(self, folds=5):
        return cross_val_score(self.model, self.train_x, self.train_y, cv=folds)

    @timed
    def fit(self):
        print("Fitting model")
        self.model.fit(self.train_x, self.train_y)

    @timed
    def predict(self):
        print("Making predictions")
        self.predictions = self.model.predict(self.test_x)

    @timed
    def write_submission(self, prefix):
        output_path = path.join(self.working_directory, 'submission_{}.csv'.format(prefix))
        print("Writing submission to file:", output_path)
        write_predictions(self.predictions, output_path)

    def prepare_submission(self, suffix=None):
        print("Preparing submission for model: ", self.model_name)
        if not suffix:
            suffix = self.model_name
        if self.full_data_loaded:
            print("Full training set already loaded.")
        else:
            print("Will load full training set.")
            self.load_train_data()
        self.fit()
        if self.test_x:
            print("Test set already loaded.")
        else:
            self.load_test_data()
        self.predict()
        self.write_submission(suffix)
        print("Finished preparing submission for model: ", self.model_name)
