import json

import cv2
import keras.backend as K
import numpy as np
from keras.models import load_model
from picasso.models.base import BaseModel

from ml.image.classifier.keras import ImageKerasClassifier
from ml.image.experiment.registry import EXPERIMENT_REGISTRY
from ml.image.transformer.image import ImageTransformer


class KerasModel(BaseModel):
    NUM_TOP_PREDICTIONS = 4

    """
    Implements model loading functions for Keras.
    Using this Keras module will require the h5py library, which is not
    included with Keras.
    """

    def __init__(self, *args, **kwargs):
        self._class_map_path = None
        self._experiment_name = None

        self._class_map = None
        self._reverse_class_map = None
        self._experiment_cls = None
        self._classifier = None
        self._width = None
        self._height = None
        self._num_channels = None
        super(KerasModel).__init__(*args, **kwargs)

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = ImageKerasClassifier(
                transformer=self.transformer,
                width=self.width,
                height=self.height,
                num_channels=self.num_channels
            )
        return self._classifier

    @property
    def class_map(self):
        if self._class_map is None:
            with open(self._class_map_path) as f:
                self._class_map = json.load(f)
        return self._class_map

    @property
    def reverse_class_map(self):
        if self._reverse_class_map is None:
            self._reverse_class_map = {}
            for tag, index in self.class_map.items():
                self._reverse_class_map[index] = tag
        return self._reverse_class_map

    @property
    def experiment_cls(self):
        if self._experiment_cls is None:
            self._experiment_cls = EXPERIMENT_REGISTRY[self._experiment_name]
        return self._experiment_cls

    @property
    def width(self):
        if self._width is None:
            self._width = self.experiment_cls.width
        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = self.experiment_cls.height
        return self._height

    @property
    def num_channels(self):
        if self._num_channels is None:
            self._num_channels = self.experiment_cls.num_channels
        return self._num_channels

    def preprocess(self, raw_inputs):
        """
        Args:
            raw_inputs (list of Images): a list of PIL Image objects
        Returns:
            array (float32): num images * height * width * num channels
        """
        X = []
        for raw_im in raw_inputs:
            raw_im = cv2.cvtColor(np.array(raw_im), cv2.COLOR_RGB2BGR)
            X.append(self.classifier.transformer.transform_X(raw_im))

        X = self.classifier.format_X(X)
        return X

    def load(self, class_map_path, experiment_name, model_path):
        self._class_map_path = class_map_path
        self._experiment_name = experiment_name
        self._model_path = model_path
        self.transformer = ImageTransformer(width=self.width, height=self.height, num_channels=self.num_channels)
        # for tensorflow compatibility
        K.set_learning_phase(0)
        self._model = load_model(self._model_path)

        self._sess = K.get_session()
        self._tf_predict_var = self._model.outputs[0]
        self._tf_input_var = self._model.inputs[0]
        self._latest_ckpt_name = ''
        self._latest_ckpt_time = 1

    def decode_prob(self, class_probabilities):
        results = []
        for probabilites in class_probabilities:
            row = []
            for index, score in enumerate(probabilites):
                print(index, score, self.reverse_class_map)
                entries = dict(index=index, name=self.reverse_class_map[index], prob='{:.3f}'.format(score))
                row.append(entries)
            row = sorted(row, key=lambda x: float(x['prob']), reverse=True)[:self.NUM_TOP_PREDICTIONS]
            results.append(row)
        return results
