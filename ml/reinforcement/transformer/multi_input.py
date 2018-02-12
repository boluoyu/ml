import numpy as np

from ml.common.transformer.base import Transformer



class MultiInputTransformer(Transformer):
    def format_X(self, X):
        formatted_X = []
        for X_component in X:
            formatted_X.append(self._format_X_component(X_component))

        return formatted_X

    def _format_X_component(self, X_component):
        num_samples = len(X_component)
        num_features = len(X_component[0])

        return np.reshape(X_component, [num_samples, num_features])


