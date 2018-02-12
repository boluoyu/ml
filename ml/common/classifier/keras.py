import keras

from ml.common.classifier.base import Classifier


class KerasClassifier(Classifier):
    name = 'keras'

    def load(self, model_file_path):
        return keras.models.load_model(model_file_path, compile=True)

    def save(self, model_file_path):
        keras.models.save_model(self.model, model_file_path, overwrite=True)

    def fit(self, X, y, **kwargs):
        X = self.transform_X(X)
        X = self.format_X(X)
        y = self.format_y(y)
        model_history = self.model.fit(X, y, **kwargs)
        return model_history.history['loss'][-1]

    def compute_loss(self, X, y):
        X = self.transform_X(X)
        X = self.format_X(X)
        y = self.format_y(y)
        return self.model.evaluate(X, y)

    def predict(self, X):
        X = self.transform_X(X)
        X = self.format_X(X)
        predictions = self.model.predict(X)
        return predictions
