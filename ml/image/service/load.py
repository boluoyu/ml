import cv2


class ImageLoader:
    flag = cv2.IMREAD_COLOR

    def load(self, image_file_path):
        return cv2.imread(image_file_path, self.flag)
