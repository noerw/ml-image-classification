import numpy as np
from PIL import Image

'''
This file serves as a template outlining the MLP training and evaluation steps.

To be filled in as we progress in this project.
'''


def loadTrainingData(rgbImg, classMaskImg, trainValidateTestSplit);
    # load input images

    # create feature vectors (see featurevector.py)

    # split features into train, validate, test
    # return training, validateion, testing




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



SOURCE_IMG   = './data/train_source.png'
TRAINING_IMG = './data/train_classes.png'
TESTING_IMG  = './data/test.png'
DATA_SPLIT = [0.5, 0.25, 0.25] # percentage of train, validate, test

dataTrain, dataValidation, dataTesting = loadTrainingData(SOURCE_IMG, TRAINING_IMG, DATA_SPLIT)

model = training(dataTrain, dataValidation, dataTesting)

testing(model, TESTING_IMG)

