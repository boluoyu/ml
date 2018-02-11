from ml.common.classifier.tensorflow import TensorflowClassifier


class ImageTensorflowClassifier(TensorflowClassifier):
    name = 'tensorflow_image'

    def __init__(self, transformer, width, height, num_channels, model=None, model_file_path=None):
        self.width = width
        self.height = height
        self.num_channels = num_channels

        super().__init__(transformer, model, model_file_path)
