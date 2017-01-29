import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps
from os import path
from sklearn import cross_validation


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        func(*args, **kwargs)
        print("Execution time:", (datetime.now() - t0))
    return wrapper


def _get_data(path):
    print("Reading in data from {}".format(path))
    df = pd.read_csv(path)
    print("Done reading.")
    return df.as_matrix()


def get_train_data(path='/Volumes/HD-LBU2/kaggle/mnist/train.csv', limit=None):
    data = _get_data(path)
    np.random.shuffle(data)
    x = data[:, 1:] / 255.0  # data is from 0..255
    y = data[:, 0]
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


def get_test_data(path='/Volumes/HD-LBU2/kaggle/mnist/test.csv'):
    return _get_data(path)


def write_predictions(predictions, path='/Volumes/HD-LBU2/kaggle/mnist/submission.csv'):
    df = pd.DataFrame({
        "ImageId": list(range(1, len(predictions)+1)),
        "Label": predictions
    })
    df.to_csv(path, index=False, header=True)


def show_figure(x):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()


class ModelTester(object):

    def __init__(self, wd):
        self.working_directory = wd
        self.train_x = None
        self.train_y = None
        self.model = None

    @timed
    def load_data(self, limit=None):
        self.train_x, self.train_y = get_train_data(path=path.join(self.working_directory, 'train.csv'), limit=limit)
        return self.train_x, self.train_y

    def set_model(self, m):
        self.model = m

    @timed
    def cross_validate(self, folds=5):
        # Get cross-validation scores
        return cross_validation.cross_val_score(self.model, self.train_x, self.train_y, cv=folds)


