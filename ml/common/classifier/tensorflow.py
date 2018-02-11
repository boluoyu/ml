import tensorflow as tf

from ml.common.classifier.base import Classifier


class TensorflowModel:
    def __init__(self, session, loss, saver, optimizer, input, targets, predictions):
        self.saver = saver
        self.session = session
        self.loss = loss
        self.optimizer = optimizer
        self.input = input
        self.targets = targets
        self.predictions = predictions

        self.session.run(tf.global_variables_initializer())


class TensorflowClassifier(Classifier):
    name = 'tensorflow'

    def load(self, model_file_path):
        self.model.saver.restore(self.model.session, model_file_path)

    def save(self, model_file_path):
        self.model.saver.save(self.model.session, model_file_path)

    def fit(self, X, y, **kwargs):
        X = list(map(self.transformer.transform, X))
        X = self.format_X(X)
        y = self.format_y(y)

        loss, optimizer = self.model.session.run(
            fetches=[self.model.loss, self.model.optimizer],
            feed_dict={
                self.model.input: X,
                self.model.targets: y
            }
        )

        return loss

    def predict(self, X):
        X = list(map(self.transformer.transform, X))
        X = self.format_X(X)

        predictions = self.model.session.run(
            fetches=self.model.predictions,
            feed_dict={
                self.model.input: X
            }
        )

        return predictions[0]

    def compute_loss(self, X, y):
        X = list(map(self.transformer.transform, X))
        X = self.format_X(X)
        y = self.format_y(y)

        return self.model.session.run(
            fetches=self.model.loss,
            feed_dict={
                self.model.input: X,
                self.model.targets: y
            }
        )
