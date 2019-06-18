from os import listdir

import numpy as np
from PIL import Image

# maps class names to feature vector
CLASSES = {
    'river': [1, 0, 0],
    'sky':   [0, 1, 0],
    'other': [0, 0, 1],
}

PIXEL_DISTANCE = 5 # number of pixels to consider as neighbours

def getIntensity(pixel):
    ''' transform pixel to value
    '''
    if typeof pixel == list: # RGB
        return (pixel[0] + pixel[1] + pixel[2]) / 3.0 / 255.0
    else: # grayscale
        return pixel / 255.0


def featureVectorHenry(image, klasse, pDistance):
    '''
    moving window:
    extracts pDistance neighbouring pixels in each direction
    '''

    pixels = np.array(image)
    width, height = image.size
    all_pixels = []
    for x in range(width-pDistance)[pDistance::pDistance*2]:
        for y in range(height-pDistance)[pDistance:]:
            px = [getIntensity(pixels[x, y])]

            for z in range(pDistance+1)[1:]:
                px += [
                    getIntensity(pixels[x-z, y]),
                    getIntensity(pixels[x+z, y]),
                    getIntensity(pixels[x, y-z]),
                    getIntensity(pixels[x, y+z]),
                ]

            px += klasse
            all_pixels.append(px)

    return all_pixels


def featureVectorFromImage(image, className):
    '''
    moving window across image.
    feature vector:
    - pixel (self)
    - 10 neighbours X axis
    - 10 neighbours y axis
    - 3 values for image class as [river, sky, other]
    '''

    return featureVectorHenry(image, CLASSES[className], PIXEL_DISTANCE)



##### MAIN#####

baseDir = './project/trainingdata'
features = [] # will contain the feature vectors for all training pixels


# load images from each class subfolder
for className in CLASSES.keys():
    for fileName in listdir('{}/{}'.format(baseDir, className)):
        filepath = '{}/{}/{}'.format(baseDir, className, fileName)
        print('processing', filepath)

        image = Image.open(filepath).convert('L') # as grayscale
        features += featureVectorFromImage(image, className)

print('resulting feature vector')
features = np.array(features)
print(np.shape(features))
print(features)
