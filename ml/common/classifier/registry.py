from ml.common.classifier.keras import KerasClassifier
from ml.common.classifier.tensorflow import TensorflowClassifier

CLASSIFIER_REGISTRY = {
    KerasClassifier.name: KerasClassifier,
    TensorflowClassifier.name: TensorflowClassifier
}
