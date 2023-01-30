
import numpy as np
import cv2 as cv
from easyocr import Reader

class Ocr:

    def __init__(self):
        self.reader = Reader(['en'], gpu = True)

    def recognize(self, image: np.array, box: list) -> str:
        """
        :param image: image array
        :param box: list with coords: XYXY
        :return: recognized text
        """
        # Preprocessing
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur_image = cv.medianBlur(gray_image, 3)
        crop_image = blur_image[box[1] : box[3], box[0] : box[2]]

        # OCR
        text_ocr = self.reader.readtext(crop_image)
        text_string = ''
        for text_block in text_ocr:
            text_string += text_block[1]
        return text_string

