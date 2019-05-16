import numpy as np
from PIL import Image

from utils import (
    processImagesFromCli,
    pseudoDownsample,
)

def processImage(image):
    return Image.fromarray(pseudoDownsample(image, (30, 30)))

processImagesFromCli(processImage, 'rgb_resample')
