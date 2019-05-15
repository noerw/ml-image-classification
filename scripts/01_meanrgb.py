import numpy as np
from PIL import Image

from utils import (
    processImagesFromCli,
    meanByChannel,
    applyRgbToRegion,
)

def meanRgb1(image):
    img = np.array(image) # 3D array: rows of columns of channels
    avgRgb = meanByChannel(img)
    print('average rgb: {}'.format(avgRgb))

    applyRgbToRegion(img, avgRgb, (0, 0, 40, 40))
    return Image.fromarray(img)

processImagesFromCli(meanRgb1, 'rgb')
