import numpy as np
from PIL import ImageFont, ImageDraw, Image


def draw_boxes(boxesList, imageArray, color="yellow", solid_box=False):
    """
    Draw object box on the image
    :param boxesList: list with image boxes dicts:
            'bbox': list with boxes coords XYXY (int)
            'class_id': int, class indexes from 0
            'class_name': int, class name
            'score': float, box score
    :param imageArray: np.array with image
    :param objectNames: list with object names
    :param color: tuple with RGB color
    :param solid_box: make object box solid or not
    :return: np.array: corrected image
    """
    font_size = 15
    font = ImageFont.truetype("/etc/alternatives/fonts-japanese-mincho.ttf", font_size)
    _, descent = font.getmetrics()
    image = Image.fromarray(imageArray[:, :, ::-1])
    borderColor = color
    for box in boxesList:
        left = int(box["bbox"][0])
        top = int(box["bbox"][1])
        right = int(box["bbox"][2])
        bottom = int(box["bbox"][3])
        boxText = f"{box['class_name']}_{box['score']:.3f}_{box['text']}"

        draw = ImageDraw.Draw(image)
        textWidth = font.getmask(boxText).getbbox()[2]
        textHeight = font.getmask(boxText).getbbox()[3] + descent
        draw.rectangle(
            [(left, top - font_size), (left + textWidth, top - font_size + textHeight)],
            fill=borderColor,
            outline=borderColor,
        )
        draw.text((left, top - font_size), boxText, font=font, fill="black")
        fill_style = borderColor if solid_box else None
        draw.rectangle(
            [(left, top), (right, bottom)], fill=fill_style, outline=borderColor
        )

    imageArray = np.array(image)[:, :, ::-1]
    return imageArray


def coords_scale_to_abs(scale_coords, image_width, image_height):
    """
    Convert Yolo coords (X, Y, X, Y) float(0,1) to absolute coords (X, Y, X, Y) int
    scale_coords: (center_x, center_y, width, height)
    return: list abs coords:
        (left, top, right, bottom)
    """
    left, top, right, bottom = scale_coords
    left = int(left * image_width)
    right = int(right * image_width)
    top = int(top * image_height)
    bottom = int(bottom * image_height)

    if left < 0:
        left = 0
    if right >= image_width:
        right = image_width - 1
    if top < 0:
        top = 0
    if bottom >= image_height:
        bottom = image_height - 1

    return (left, top, right, bottom)
