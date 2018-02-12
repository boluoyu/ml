from abc import abstractmethod


class Classifier:
    def __init__(self, model=None, model_file_path=None, transformer=None):
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
        if not self.transformer:
            return X

        X = self.transformer.format_X(X)
        return X

    def format_y(self, y):
        if not self.transformer:
            return y

        y = self.transformer.format_y(y)
        return y

    def transform_y(self, y):
        if not self.transformer:
            return y

        y = self.transformer.transform_y(y)
        return y

    def transform_X(self, X):
        if not self.transformer:
            return X

        X = self.transformer.transform_X(X)
        return X


