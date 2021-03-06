import cv2
import numpy as np

from ml.common.transformer.base import Transformer


class ImageTransformer(Transformer):
    name = 'image_transformer'

    def __init__(self, width, height, num_channels):
        self.width = width
        self.height = height
        self.num_channels = num_channels

    def transform_X(self, X):
        num_samples = len(X)

        X = list(map(self._transform_x, X))
        return np.reshape(X, [num_samples, self.width, self.height, self.num_channels])

    def format_X(self, X):
        num_samples = len(X)

        return np.reshape(X, [num_samples, self.width, self.height, self.num_channels])

    def _transform_x(self, x):
        x_type = type(x)
        x = np.array(x)
        #x = cv2.imdecode(x, flags=cv2.IMREAD_GRAYSCALE)
        x = self.resize(x, target_width=self.width, target_height=self.height).astype(np.float32)
        x = x / 255
        return x

    def transform_y(self, y):
        return y

    def format_y(self, y):
        return y

    def resize(self, image, target_width, target_height):
        try:
            width, height = image.shape[1], image.shape[0]
        except Exception:
            print('ERROR', image, image.shape)
            raise ValueError()
        if width != target_width or height != target_height:
            image = cv2.resize(image, tuple([target_height, target_height]), interpolation=cv2.INTER_AREA)

        return image

    """
    def resize(image, height, width, size=DOWNSAMPLE_DIMENSIONS):
        import cv2

        if width > height:
            scale = float(size) / width
        else:
            scale = float(size) / height

        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized
    """
