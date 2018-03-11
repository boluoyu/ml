import cv2
import numpy as np

from ml.common.transformer.base import Transformer


class AutoencoderTransformer(Transformer):
    name = 'image_transformer'

    def __init__(self, width, height, num_channels):
        self.width = width
        self.height = height
        self.num_channels = num_channels

    def transform_X(self, X):
        X = list(map(self._transform_x, X))
        return X

    def format_X(self, X):
        num_samples = len(X)
        return np.reshape(X, [num_samples, self.width, self.height, self.num_channels])

    def _transform_x(self, x):
        x_type = type(x)
        x = np.array(x, dtype=np.uint8)
        print(x_type, x.shape)
        x = self.resize(x, target_width=self.width, target_height=self.height).astype(np.float32)
        x = x / 255
        return x

    def greyscale(self, image):
        grey = np.zeros((image.shape[0], image.shape[1]))  # init 2D numpy array
        # get row number
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])

        return grey

    def transform_y(self, y):
        y['decoder'] = self.transform_X(y['decoder'])
        return y

    def format_y(self, y):
        y['decoder'] = self.format_X(y['decoder'])
        y['classifier'] = np.array(y['classifier'])
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
