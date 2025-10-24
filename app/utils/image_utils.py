from uuid import uuid4
import cv2
import numpy as np
import os

from app.core.config import settings


def load_image_from_buffer(contents):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

def convert_image_bool_to_uint(image):
    return image.astype(np.uint8) * 255

def get_mask_path():
    return os.path.join(settings.MASK_PATH, f"{uuid4().hex}_maks.png")

def denormalize_image(image):
    img_min = image.min()
    img_max = image.max()

    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)
    
    image = (image - img_min) / (img_max - img_min)
    return (image * 255).astype(np.uint8)