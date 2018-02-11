from ml.common.classifier.keras import KerasClassifier


class ImageKerasClassifier(KerasClassifier):
    name = 'keras_image'

    def __init__(self, transformer, width, height, num_channels, model=None, model_file_path=None):
        self.width = width
        self.height = height
        self.num_channels = num_channels

        super().__init__(transformer, model, model_file_path)
