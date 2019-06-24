from builtins import type

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
    if type(pixel) == tuple or type(pixel) == list: # RGB
        return (pixel[0] + pixel[1] + pixel[2]) / 3.0 / 255.0
    else: # grayscale
        return pixel / 255.0


def featureVectorHenry(image, classified_image, pDistance):
    '''
    moving window:
    extracts pDistance neighbouring pixels in each direction
    '''

    features = {
        classname: [] for classname in CLASSES.keys()
    }

    classified_pixels =  np.array(classified_image).astype(np.float) / 255.0 
    pixels = image.load() # note: x and y are in a different order than in 'classified_pixels'
    width, height = image.size

    for x in range(width-pDistance)[pDistance::]:
        for y in range(height-pDistance)[pDistance::]:
            px = [getIntensity(pixels[x, y])]

            for z in range(pDistance+1)[1:]:
                px += [
                    getIntensity(pixels[x-z, y]),
                    getIntensity(pixels[x+z, y]),
                    getIntensity(pixels[x, y-z]),
                    getIntensity(pixels[x, y+z]),
                ]
            px += list(classified_pixels[y,x]) # append class identification
            for name, classs in CLASSES.items():
                if classs == list(classified_pixels[y,x]):
                    features[name].append(px) # include 
    return features


def featureVectorFromImage(original_image, classified_image):
    '''
    moving window across image.
    feature vector:
    - pixel (self)
    - 10 neighbours X axis
    - 10 neighbours y axis
    - 3 values for image class as [river, sky, other]
    '''

    return featureVectorHenry(original_image, classified_image, PIXEL_DISTANCE)



##### MAIN#####


def loadSplitFeatures(baseDir):
    featuresSky = [] # will contain the feature vectors for all training pixels
    featuresRiver = []
    featuresOther = []
    featuresTraining = []
    featuresTesting = []
    featuresValidation = []


    # load classified and original image
    classified_image = Image.open("./data/drone150meter_classified.png")
    original_image = Image.open("./data/drone150meter.jpg").convert('L')
    features = featureVectorFromImage(original_image, classified_image)


    #TODO: Split features into testing, training and validation

    featuresTraining += featuresSky[0:int(len(featuresSky)/3)]
    featuresTraining += featuresRiver[0:int(len(featuresRiver)/3)]
    featuresTraining += featuresOther[0:int(len(featuresOther) / 3)]
    featuresTesting += featuresSky[int(len(featuresSky)/3):int(len(featuresSky)/3*2)]
    featuresTesting += featuresRiver[int(len(featuresRiver)/3):int(len(featuresRiver)/3*2)]
    featuresTesting += featuresOther[int(len(featuresOther) / 3):int(len(featuresOther) / 3*2)]
    featuresValidation += featuresSky[int(len(featuresSky)/3*2):int(len(featuresSky))]
    featuresValidation += featuresRiver[int(len(featuresRiver)/3*2):int(len(featuresRiver))]
    featuresValidation += featuresOther[int(len(featuresOther) / 3*2):int(len(featuresOther))]

    return np.array(featuresTraining), np.array(featuresValidation), np.array(featuresTesting)

# print('creating featurevector')
# features, validation, testing = loadSplitFeatures('./trainingdata')
# print('resulting feature vector')
# features = np.array(features)
# print(np.shape(features))
# print(features)
