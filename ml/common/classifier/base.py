import numpy as np

from abc import abstractmethod


class Classifier:
    def __init__(self, transformer, model=None, model_file_path=None):
        self.model = model
        self.model_file_path = model_file_path
        self.transformer = transformer

    @abstractmethod
    def save(self, model_file_path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_file_path):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    def format_X(self, X):
        num_samples = len(X)
        return np.reshape(X, [num_samples, self.height, self.width, self.num_channels])

    def format_y(self, y):
        num_samples = len(y)
        num_targets = len(y[0])
        return np.reshape(y, [num_samples, num_targets])
