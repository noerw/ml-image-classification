import numpy as np
from PIL import Image

from utils import (
    processImagesFromCli,
    meanByChannel,
    applyRgbToRegion,
    subimages,
)

def processImage(image):
    img = np.array(image)
    for subimg, box in subimages(image, sizePixels=(30, 30)):
        avgRgb = meanByChannel(subimg)
        applyRgbToRegion(img, avgRgb, box)

    return Image.fromarray(img)

processImagesFromCli(processImage, 'rgb_resample')
