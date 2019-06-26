import os
import numpy as np
from PIL import Image
from marsland_example import mlp

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

    if classified_image:
        classified_pixels = np.array(classified_image).astype(np.float) / 255.0

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

            if classified_image:
                px += list(classified_pixels[y,x]) # append class identification
            features.append(px)

    return features

def balanceClasses(all_features, numClasses):
    all_features = np.array(all_features)

    # split features by classes
    columns = np.shape(all_features)[1]
    featuresByClass = [None] * numClasses # list of np.arrays, one per class

    for i in range(numClasses):
        indices = np.where(all_features[:,columns - i - 1] == 1)
        featuresByClass[i] = all_features[indices]

    maxClassSize = 0
    for features in featuresByClass:
        if len(features) > maxClassSize:
            maxClassSize = len(features)


    # make all classess approximately the length of maxClassSize
    # by appending itself repeatetly
    for features in featuresByClass:
        # if balanceFactor == 0   --> has desired size
        # if balanceFactor == 1.6 --> needs 160% size
        balanceFactor = float(maxClassSize) / len(features) - 1.0
        assert balanceFactor >= 0

        f = list(features)
        while balanceFactor > 0:
            if balanceFactor >= 1: # full length needs to be appended
                features = np.append(features, f, axis=0)
                balanceFactor -= 1
            else: # subset needs to be added
                numFeatures = int(len(f) * balanceFactor)
                features = np.append(features, f[0:numFeatures], axis=0)
                balanceFactor = 0

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
    print('----> opening images')
    img = Image.open(rgbImg).convert('L') # as grayscale
    classMask = Image.open(classMaskImg).convert('RGB')

    print('----> extracting feature vectors')
    features = featureVector(img, 5, classMask)

    print('----> balancing classes')
    features = balanceClasses(features, NUM_CLASSES)

    # split features into train, validate, test & return
    print('----> splitting data into train/validate/test sets')
    return splitFeatures(features, trainValidateTestSplit)









def training(data_train, data_validate, data_test):
    ''' train & validate
    '''

    # split targets from feature vector
    number_predictors = len(data_train[0]) - NUM_CLASSES # 3 classes / output neurons

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
        0.1, 1000
    )

    model.confmat(test_inputs, test_targets) # Confusion matrix


    return model






def testing(model, imageLocation, OUTPUT_IMG, pDistance=5):
    ''' classify another image, and show the classification result as image
    '''
    img2 = Image.open(imageLocation)
    img = Image.open(imageLocation).convert('L')
    img_width = img.width
    img_height = img.height
    arr = np.array(img2)
    inputs = featureVector(img)
    inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
    outputs = model.mlpfwd(inputs)
    outputs = np.argmax(outputs,1)
    for m in range(img_height-pDistance*2):
        for n in range(img_width-pDistance*2):
            classifiedPixel = [0,0,0]
            index = m*(img_width-pDistance*2)+n
            classifiedPixel[outputs[index]]=255
            arr[m+pDistance,n+pDistance] = classifiedPixel
    img_out = Image.fromarray(arr)
    img_out.show()
    img_out.save(OUTPUT_IMG)



pwd = os.path.dirname(os.path.realpath(__file__))

SOURCE_IMG = pwd + '/../data/drone150meter.jpg'
CLASS_IMG  = pwd + '/../data/drone150meter_classified.png'
# SOURCE_IMG = pwd + '/../data/drone150meter_small.png'           # NOTE: for faster testing purposes only!
# CLASS_IMG  = pwd + '/../data/drone150meter_classified_small.png'
OUTPUT_IMG = pwd + '/test.png'
DATA_SPLIT = [0.5, 0.25, 0.25] # percentage of train, validate, test
NUM_CLASSES = 3

dataTrain, dataValidation, dataTesting = loadTrainingData(SOURCE_IMG, CLASS_IMG, DATA_SPLIT)

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
