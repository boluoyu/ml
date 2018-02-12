import numpy as np

from ml.common.classifier.keras import KerasClassifier


class ImageKerasClassifier(KerasClassifier):
    name = 'keras_image'

    def __init__(self, transformer, width, height, num_channels, model=None, model_file_path=None):
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.transformer = transformer

        super().__init__(model, model_file_path)

    def format_X(self, X):
        num_samples = len(X)
        return np.reshape(X, [num_samples, self.height, self.width, self.num_channels])

    def transform_X(self, X):
        return list(map(self.transformer.transform, X))
