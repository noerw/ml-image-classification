import os
import numpy as np
from PIL import Image
from marsland_example import mlp

scriptDir = os.path.dirname(__file__) or '.'
trainingdata = scriptDir + '/trainingdata'

'''
This file serves as a template outlining the MLP training and evaluation steps.

To be filled in as we progress in this project.
'''

def featureVector(image, pDistance=5, classified_image=None):
    '''
    moving window across image.
    feature vector:
    - pixel (self)
    - pDistance * 2 neighbours X axis
    - pDistance * 2 neighbours y axis
    - optional: 3 values for image class as [R,G,B]
    '''
    
    features = []

    if(classified_image not None):
        classified_pixels =  np.array(classified_image).astype(np.float) / 255.0 
    
    pixels = image.load() # note: x and y are in a different order than in 'classified_pixels'
    width, height = image.size

    for x in range(width-pDistance)[pDistance::]:
        for y in range(height-pDistance)[pDistance::]:
            px = [pixels[x, y] / 255.0]

            for z in range(pDistance+1)[1:]:
                px += [
                    pixels[x-z, y] / 255.0,
                    pixels[x+z, y] / 255.0,
                    pixels[x, y-z] / 255.0,
                    pixels[x, y+z] / 255.0,
                ]
            if(classified_image not None):
                px += list(classified_pixels[y,x]) # append class identification
            features.append(px) # include 
    return features

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
    data_train, data_test, data_validate = loadSplitFeatures(trainingdata)

    # split targets from feature vector
    number_predictors = len(data_train[0]) - 3 # 3 classes / output neurons

    inputs = data_train[:,0:number_predictors]
    targets = data_train[:,number_predictors:]

    validation_inputs = data_validate[:,0:number_predictors]
    validation_targets = data_validate[:,number_predictors:]

    test_inputs = data_test[:,0:number_predictors]
    test_targets = data_test[:,number_predictors:]

    # initialize model
    model = mlp.mlp(inputs, targets, 5)

    # train until validation says we're not learning anymore
    model.earlystopping(
        inputs, targets,
        validation_inputs, validation_targets,
        0.1,
    )

    #net.confmat(data_train[:,0:number_predictors], data_train[:,number_predictors:]) # Confusion matrix


    return model






def testing(model, imageLocation, OUTPUT_IMG):
    ''' classify another image, and show the classification result as image
    '''
    img = Image.open(imageLocation)
    img_width = img.width
    img_height = img.height
    arr = np.array(img)
    inputs = featureVector(img)
    inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
    outputs = model.mlpfwd(inputs)
    outputs = np.argmax(outputs,1)
    for m in range(img_height-pDistance)[pDistance::]:
        for n in range(img_width-pDistance)[pDistance:]:
            classifiedPixel = [0,0,0]   
            classifiedPixel[outputs[((m-5)*img_width)+n-5]]=255
            arr[m,n] = classifiedPixel
    img_out = Image.fromarray(arr).show()
    img_out.save(OUTPUT_IMG)


    pass



pwd = os.path.dirname(os.path.realpath(__file__))

# SOURCE_IMG = pwd + '/../data/drone150meter.jpg'
# CLASS_IMG  = pwd + '/../data/drone150meter_classified.png'
SOURCE_IMG = pwd + '/../data/drone150meter_small.png'           # NOTE: for faster testing purposes only!
CLASS_IMG  = pwd + '/../data/drone150meter_classified_small.png'
OUTPUT_IMG = pwd + '/test.png'
DATA_SPLIT = [0.5, 0.25, 0.25] # percentage of train, validate, test

dataTrain, dataValidation, dataTesting = loadTrainingData(SOURCE_IMG, CLASS_IMG, DATA_SPLIT)
#dataTrain, dataValidation, dataTesting = loadTrainingData(SOURCE_IMG, TRAINING_IMG, DATA_SPLIT)

print('test, validate, testing')
print(
    np.shape(dataTrain),
    np.shape(dataValidation),
    np.shape(dataTesting),
)
model = training(dataTrain, dataValidation, dataTesting)

testing(model, SOURCE_IMG, OUTPUT_IMG)

# model = training(dataTrain, dataValidation, dataTesting)

# testing(model, OUTPUT_IMG)

