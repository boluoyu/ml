import cv2
import numpy as np

from ml.image.transform.base import Transformer


class ImageTransformer(Transformer):
    name = 'image_transformer'

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def transform(self, image):
        image = self.resize(image, target_width=self.width, target_height=self.height).astype(np.float32)
        return image

    def resize(self, image, target_width, target_height):
        width, height = image.shape[1], image.shape[0]
        if width != target_width or height != target_height:
            image = cv2.resize(image, tuple([target_height, target_height]), interpolation=cv2.INTER_AREA)

        return image
