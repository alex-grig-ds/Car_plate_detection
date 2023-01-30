
from pathlib import Path
import time as tm
import traceback as tb

from tqdm import tqdm
import torch
import cv2 as cv
import pandas as pd
import click

from src.class_dataset import Dataset
from src.class_yolo_detector import YoloDetector
from src.class_ocr import Ocr
from src.utils import draw_boxes
from src.app_logger import logger
from config import *

@click.command()
@click.option('--input_folder', '-if', default = './input',
              type=click.Path(exists=True), required=True,
              help = 'Folder with checking images.')
@click.option('--output_folder', '-of', default = './result',
              type=click.Path(), required=True,
              help = 'Folder for detection results.')
@click.option('--saved_objects', '-so', default = './detected_objects.csv',
              type=click.Path(), required=True,
              help = 'CSV file for detection results saving.')
def plate_detection(input_folder: str, output_folder: str, saved_objects: str):
    img_folder = Path(input_folder)
    result_folder = Path(output_folder)
    result_folder.mkdir(parents=True, exist_ok=True)
    if DRAW_BOXES:
        for file in result_folder.glob("*.*"):
            file.unlink()

    args = {'folder': img_folder.resolve(), 'qnty': 0}
    dataset = Dataset('folder', **args)

    if torch.cuda.is_available():
        logger.info(f'CUDA is available. CUDA version: {torch.version.cuda}')
        device_idx = torch.cuda.current_device()
        device = torch.device(f'cuda:{device_idx}')
    else:
        logger.info(f'Start search with CPU.')
        device = torch.device('cpu')
    torch.no_grad()
    detector = YoloDetector(device, PLATE_WEIGHTS, PLATE_YAML, IMG_SIZE, DETECT_CONFIDENCE, DETECT_IOU)
    recognizer = Ocr()
    logger.info("Start detection.")

    imageDf = pd.DataFrame(
        columns=['Image_file', 'Object_class', 'top', 'left', 'bottom', 'right', 'text'])
    t1 = tm.time()
    for image, imgFile in tqdm(dataset, desc='Object detection progress:'):
        try:
            currentBoxes = detector.make_detection(image)
            for box in currentBoxes:
                dfIdx = len(imageDf)
                imageDf.loc[dfIdx, 'Image_file'] = imgFile
                imageDf.loc[dfIdx, 'Object_class'] = box['class_id']
                imageDf.loc[dfIdx, 'Score'] = box['score']
                imageDf.loc[dfIdx, 'top'] = box['bbox'][1]
                imageDf.loc[dfIdx, 'left'] = box['bbox'][0]
                imageDf.loc[dfIdx, 'bottom'] = box['bbox'][3]
                imageDf.loc[dfIdx, 'right'] = box['bbox'][2]
                imageDf.loc[dfIdx, 'text'] = box['text'] = recognizer.recognize(image, box['bbox'])

            if DRAW_BOXES:
                image = draw_boxes(currentBoxes, image, 'yellow')
                if SAVE_DRAW_NEGATIVES or currentBoxes:
                    cv.imwrite(Path.joinpath(result_folder, imgFile), image)
        except:
            logger.info(f"Some errors occur with image {imgFile}. Reason: {tb.format_exc()}")

    imageDf.to_csv(saved_objects, index=False)
    logger.info(f"Detection finished. Take time: {tm.time()-t1}")


if __name__ == '__main__':
    plate_detection()
