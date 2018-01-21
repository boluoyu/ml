import tensorflow as tf

from ml.image.classifier.base import Classifier


class TensorflowClassifier(Classifier):
    name = 'tensorflow'

    def __init__(self, transformer, model=None, model_file_path=None):
        super(TensorflowClassifier, self).__init__(transformer, model, model_file_path)

        tf.reset_default_graph()
        self.session = tf.Session()

    def load(self, model_file_path):
        self.initialize_graph()
        self._initialize_saver()
        self.saver.restore(self.session, model_file_path)

    def _initialize_saver(self):
        self.saver = tf.train.Saver()

    def initialize_graph(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, x):
        raise NotImplementedError()

