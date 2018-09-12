import numpy as np
from keras import backend as K
from keras.callbacks import Callback


class GradientDebugger(Callback):
    def on_batch_end(self, batch, logs=None):
        self._log_weights(self.model)

    def _log_weights(self, model):
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                self._log_weights(layer)
            else:
                for weight in layer.weights:
                    print(layer.name, weight.name, np.mean(K.eval(weight)))