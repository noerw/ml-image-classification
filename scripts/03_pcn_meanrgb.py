import numpy as np
from PIL import Image

from pcn import Pcn
from utils import (
    processImagesFromCli,
    pseudoDownsample,
)

SIZE = (30, 30)
data = []
def loadImageData(image):
    # imgdata = np.array(image)
    # imgdata = pseudoDownsample(image, (30, 30)) # keep original resolution
    imgdata = np.array(image.resize(SIZE, Image.NEAREST)) # actual downsampling

    # flatten to list of [rgb] or [class]: the x,y coordinates aren't of interest
    if (image.mode == 'L'):
        imgdata = imgdata.reshape(-1, 1) # training mask has only one channel
        imgdata = np.where(imgdata == 0, 0, 1) # map to 0 / 1
    else:
        imgdata = imgdata.reshape(-1, imgdata.shape[-1]) # image probably has RGB
    print(imgdata.shape)

    data.append(imgdata)

print('--> reading & resampling images')
processImagesFromCli(loadImageData)
if len(data) != 2:
    raise Exception('needs exactly two images: [reference_img, training_mask]')

print('--> training Perceptron')
inputs, targets = data
pcn = Pcn(inputs, targets)
pcn.pcntrain(inputs, targets, 0.25, 1000)
pcn.confmat(inputs, targets)

print('--> preview of input forwarding')
result = pcn.pcnfwd(np.hstack((inputs, np.zeros(targets.shape))))
result = result.reshape(SIZE + (1,))
result = np.where(result == 0, 0, 255)
print(result.shape)
Image.fromarray(result, 'I;16').show()
