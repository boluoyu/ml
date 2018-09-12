from keras.models import load_model


class Classifier:
    _model = None

    def __init__(self, name, class_map):
        """
        Instantiates classifier
        :param name: name of classifier
        :param class_map: mapping of class indices to class names: {0: "fire", 1: "flood"}
        """
        self.name = name
        self.class_map = class_map

    def predict_proba(self, items):
        raise NotImplementedError()


    def load(self, model_file_path):
        raise NotImplementedError()


class KerasClassifier:
    def load(self, model_file_path):
        self._model = load_model(model_file_path, compile=True)

    def predict_proba(self, items):
        """
        :param items: np array of encoded items
        :return: predictions {"fire": 0.1, "flood": 0.2}
        """
        predictions = self._model.predict_proba(items)
        predictions = {category: float(predictions[int(ix)]) for ix, category in self.class_map.items()}
        return predictions


class Transformer:
    def transform_X(self, items):
        raise NotImplementedError()


class TextTransformer(Transformer):
    def transform_X(self, texts):
        return texts


classifier = Classifier(name="my_classifier", class_map={0: "fire", 1: "flood"})
classifier.load("my_model.h5")
transformer = Transformer()
predictions = classifier.predict_proba(transformer.transform_X(["hello world"]))
