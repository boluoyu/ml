from ml.image.classifier.keras import KerasClassifier
from ml.image.classifier.tensorflow import TensorflowClassifier

CLASSIFIER_REGISTRY = {
    KerasClassifier.name: KerasClassifier,
    TensorflowClassifier.name: TensorflowClassifier
}
