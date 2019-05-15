import sys
from os import path
from PIL import Image
import numpy as np

def processImagesFromCli(processfunc, resultPostfix = None):
    '''
    Processes each image path passed at the command line with processfunc.
    processfunc's only argument is an PIL.Image.
    If resultPostfix is provided, the resulting image is saved with the given postfix path.
    '''
    files = sys.argv[1:]
    if not len(files):
        return print('no input files provided.')

    for infile in files:
        img = Image.open(infile)
        result = processfunc(img)

        if resultPostfix:
            base, ext = path.splitext(infile)
            outfile = '{}_{}.png'.format(base, resultPostfix)
            result.save(outfile)

def meanByChannel(image):
    '''
    Returns the mean RGB value as list [r,g,b] for the given image.

    image: numpy array from PIL.Image, or PIL.Image
    '''
    return np.mean(image, axis=(0,1))

def applyRgbToRegion(image, rgb, box):
    '''
    Applies the given rgb value to all pixels in within the coordinate box.

    image: numpy array from PIL.Image, or PIL.Image
    rgb: list of [r,g,b]
    box: tuple of (xmin,ymin,xmax,ymax)
    '''
    # TODO: check dimensions first
    for row in range(box[0], box[2]):
        for col in range(box[1], box[3]):
            image[row][col] = rgb
    return image
