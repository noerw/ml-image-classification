import os
import numpy as np
from PIL import Image

from featurevector import featureVectorHenry

'''
This file serves as a template outlining the MLP training and evaluation steps.

To be filled in as we progress in this project.
'''

def balanceClasses(featuresByClass):
    maxClassSize = max(featuresByClass, key=lambda feats: np.shape(feats)[0]) # FIXME: does not what we want

    print(maxClassSize)
    for features in featuresByClass:
        balanceFactor = float(maxClassSize) / len(features)
        # TODO: append `balancefactor` percent of the list to itself

    # merge all classes into a single array
    return np.vstack(featuresByClass)

def splitFeatures(features, trainValidateTestSplit):
    if sum(trainValidateTestSplit) != 1.0:
        raise Exception('trainValidateTestSplit dont sum to 1.0!')

    totalFeats = len(features)

    # randomize feature order
    order = list(range(totalFeats))
    np.random.shuffle(order)
    features = features[order,:]

    # do split
    f_train, f_validate, f_test = (np.array(trainValidateTestSplit) * totalFeats).astype(np.int)

    d_train    = features[0 : f_train]
    d_validate = features[f_train : f_train + f_validate]
    d_test     = features[f_train + f_validate :]

    return d_train, d_validate, d_test


def loadTrainingData(rgbImg, classMaskImg, trainValidateTestSplit):
    # load input images
    print('----> opening images')
    img = Image.open(rgbImg).convert('L') # as grayscale
    classMask = Image.open(classMaskImg).convert('RGB')

    # create feature vectors (see featurevector.py)
    print('----> extracting feature vectors')
    featuresByClass = featureVectorHenry(img, classMask, 5)

    # balance classes
    print('----> balancing classes (@INCOMPLETE!)')
    features = balanceClasses(list(featuresByClass.values()))

    # split features into train, validate, test & return
    print('----> splitting data into train/validate/test sets')
    return splitFeatures(features, trainValidateTestSplit)









def training(dataTrain, dataValidation, dataTesting):
    ''' train & validate
    '''

    # split featurevectors into inputs & targets

    # initialize model
    model = Mlp(inputs, targets, ...)

    # train until validation says we're not learning anymore
    model.earlyStopping(dataTrain, dataValidation, dataTesting)


    return model






def testing(model, testingImg):
    ''' classify another image, and show the classification result as image
    '''

    # resultimage = Image.new() (???)
    # inputs = featureVectors(testingImg, None)

    # for each pixel
    #     classification = model.feedforward(inputs[pixel])
    #     resultimage[pixel] = classification

    # resultimage.show()

    pass



pwd = os.path.dirname(os.path.realpath(__file__))

# SOURCE_IMG = pwd + '/../data/drone150meter.jpg'
# CLASS_IMG  = pwd + '/../data/drone150meter_classified.png'
SOURCE_IMG = pwd + '/../data/drone150meter_small.png'           # NOTE: for faster testing purposes only!
CLASS_IMG  = pwd + '/../data/drone150meter_classified_small.png'
OUTPUT_IMG = pwd + '/test.png'
DATA_SPLIT = [0.5, 0.25, 0.25] # percentage of train, validate, test

dataTrain, dataValidation, dataTesting = loadTrainingData(SOURCE_IMG, CLASS_IMG, DATA_SPLIT)

print('test, validate, testing')
print(
    np.shape(dataTrain),
    np.shape(dataValidation),
    np.shape(dataTesting),
)


# model = training(dataTrain, dataValidation, dataTesting)

# testing(model, OUTPUT_IMG)

