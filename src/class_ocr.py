
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

        kernel = np.ones((5, 5), np.uint8)
        block_image = cv.erode(crop_image, kernel, iterations=5)

        kernel = np.ones((15, 15), np.uint8)
        block_image = cv.morphologyEx(block_image, cv.MORPH_OPEN, kernel)

        # Detect contour of car plate
        _, contours, __ = cv.findContours(block_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        fullImageArea = block_image.shape[0] * block_image.shape[1]
        maxArea = fullImageArea
        numberAttempt = 0
        while maxArea >= fullImageArea * 0.95 and numberAttempt < 3:
            # maxArea >= fullImageArea * 0.95 - it's means that current max contour is the external border of the image
            areas = [cv.contourArea(cont) for cont in contours]
            maxIdx = areas.index(max(areas))
            maxContour = contours.pop(maxIdx)
            numberAttempt += 1

        # Detect rotation angle
        rectangle = cv.minAreaRect(maxContour)
        currentAngle = rectangle[-1]
        if currentAngle < -45:
            rotateAngle = 90 + currentAngle
        else:
            rotateAngle = currentAngle

        # Rotate image
        centerPoint = (crop_image.shape[1] // 2, crop_image.shape[0] // 2)
        rotationMatrix = cv.getRotationMatrix2D(centerPoint, rotateAngle, 1.0)
        rotatedImage = cv.warpAffine(crop_image, rotationMatrix, (crop_image.shape[1], crop_image.shape[0]),
                                     flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

        # OCR
        text_ocr = self.reader.readtext(crop_image)
        text_string = ''
        for text_block in text_ocr:
            text_string += text_block[1]
        return text_string

