"""
Make one image detection with yolov5 network
"""
import yaml

import cv2 as cv
import torch

from models.yolov5.utils.general import non_max_suppression, scale_coords
from models.yolov5.utils.augmentations import letterbox
from models.yolov5.models.experimental import attempt_load
from src.utils import coords_scale_to_abs


class YoloDetector:
    """
    Yolo v5 detection class
    """

    def __init__(
        self,
        device: torch.device,
        weights: str,
        nn_yaml: str,
        image_size=640,
        conf_thresh=0.25,
        iou_thresh=0.45,
    ):
        """
        :param device: cuda device, i.e. 0 or 0,1,2,3 or cpu
        :param weights: model.pt path
        :param nn_yaml: model yaml file path
        :param image_size: image size
        :param conf_thresh: confidence threshold
        :param iou_thresh: IOU threshold
        """
        # Load model
        with open(nn_yaml, errors="ignore") as f:
            self.names = yaml.safe_load(f)["names"]  # class names
        self.image_size = image_size
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = attempt_load(weights, map_location=device)
        return

    def make_detection(self, image):
        """
        Make detection for one image in open CV format (np.array, BGR)
        :param image:
        :return: list with boxes:
            category_id: class ID from 0
            category_name: class name
            bbox: XYXY coords
            score: box confidence

        """
        img_tensor, orig_shape, tensor_shape = self.preprocess(image)
        img_predict = self.get_img_predict(img_tensor, orig_shape, tensor_shape)
        full_predict = []
        for box in img_predict:
            # box['bbox'] = coords_from_yolo(box['bbox'], image.shape[1], image.shape[0])
            box["bbox"] = coords_scale_to_abs(
                box["bbox"], image.shape[1], image.shape[0]
            )
            box["width"] = abs(int(box["bbox"][2] - box["bbox"][0]))
            box["height"] = abs(int(box["bbox"][3] - box["bbox"][1]))
            full_predict.append(box)

        return full_predict

    def preprocess(self, orig_img):
        orig_img = cv.cvtColor(orig_img, cv.COLOR_BGR2RGB)
        img1 = letterbox(orig_img, self.image_size, auto=False)[0]
        img1 = img1.transpose(2, 0, 1)
        img_tensor1 = torch.from_numpy(img1).float().to(self.device)
        img_tensor1 /= 255
        img_tensor1 = img_tensor1.unsqueeze(0)
        return img_tensor1, list(orig_img.shape), list(img_tensor1.shape)

    def postprocess(
        self,
        out,
        tensor_shape,
        orig_shape,
    ):
        out = non_max_suppression(out, self.conf_thresh, self.iou_thresh)
        img_predict = []
        for pred in out:
            pred[:, :4] = scale_coords(
                tensor_shape[2:], pred[:, :4], orig_shape
            ).round()
            box = pred[:, :4]
            box[:, 0] /= orig_shape[1]
            box[:, 1] /= orig_shape[0]
            box[:, 2] /= orig_shape[1]
            box[:, 3] /= orig_shape[0]
            for p, b in zip(pred.tolist(), box.tolist()):
                img_predict.append(
                    {"category_id": int(p[5]), "bbox": [x for x in b], "score": p[4]}
                )
        return img_predict

    def get_img_predict(self, img_tensor, orig_shape, tensor_shape):
        """
        Return:
            category_id: class ID from 0
            class_name: class name
            bbox: center_x, center_y, width, height in relative coords (0,1)
            score: box confidence
        """
        with torch.no_grad():
            out, _ = self.model(img_tensor, augment=False)
        img_predict = self.postprocess(out, tensor_shape, orig_shape)
        for found_box in img_predict:
            found_box["class_id"] = found_box["category_id"]
            found_box.pop("category_id")
            found_box["class_name"] = self.names[found_box["class_id"]]
        return img_predict
