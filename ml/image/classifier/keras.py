import keras

from ml.image.classifier.base import Classifier


class KerasClassifier(Classifier):
    name = 'keras'

    def load(self, model_file_path):
        return keras.models.load_model(model_file_path, compile=True)

    def save(self, model_file_path):
        keras.models.save_model(self.model, model_file_path, overwrite=True)

    def fit(self, X, y, **kwargs):
        model_history = self.model.fit(X, y, **kwargs)
        return model_history.history['loss'][-1]

    def compute_loss(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        X = self.transformer.transform(X)
        predictions = self.model.predict(X)
        return predictions
