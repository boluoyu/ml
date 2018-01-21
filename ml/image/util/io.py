import cv2
import numpy as np

from io import StringIO


class DecodeImageError(Exception):
    name = 'decode_image'


def decode_image(content, cv_flags=cv2.IMREAD_COLOR):
    try:
        #stream = StringIO(content)
        bitmap = np.asarray(contentz, dtype=np.uint8)
        image = cv2.imdecode(bitmap, cv_flags)
        return image
    except cv2.error:
        raise DecodeImageError()
