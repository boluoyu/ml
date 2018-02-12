import numpy as np


class Transformer:
    def transform_X(self, X):
        return X

    def transform_y(self, y):
        return y

    def format_X(self, X):
        num_samples = len(X)
        num_features = len(X[0])

        return np.reshape(X, [num_samples, num_features])

    def format_y(self, y):
        num_samples = len(y)
        num_targets = len(y[0])

        return np.reshape(y, [num_samples, num_targets])
