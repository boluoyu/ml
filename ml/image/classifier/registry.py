from ml.image.classifier.keras import ImageKerasClassifier
from ml.image.classifier.tensorflow import ImageTensorflowClassifier

CLASSIFIER_REGISTRY = {
    ImageKerasClassifier.name:      ImageKerasClassifier,
    ImageTensorflowClassifier.name: ImageTensorflowClassifier
}
