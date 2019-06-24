import sys, os
from PIL import Image
import numpy as np
from marsland_example import mlp
from featurevector import loadSplitFeatures

scriptDir = os.path.dirname(__file__) or '.'
baseDir = scriptDir + '/trainingdata'

data_train, data_test, data_validate = loadSplitFeatures(baseDir)

# split targets from feature vector
number_predictors = len(data_train[0]) - 3 # 3 classes / output neurons

inputs = data_train[:,0:number_predictors]
targets = data_train[:,number_predictors:]

validation_inputs = data_validate[:,0:number_predictors]
validation_targets = data_validate[:,number_predictors:]

test_inputs = data_test[:,0:number_predictors]
test_targets = data_test[:,number_predictors:]
print(np.shape(inputs))
print(np.shape(targets))


net = mlp.mlp(inputs, targets, 5)
net.earlystopping(
    inputs, targets,
    validation_inputs, validation_targets,
    0.1,
)
# print(net)

net.confmat(data_train[:,0:number_predictors], data_train[:,number_predictors:])


# FIXME: check out why 22 features are needed here not 21??
prediction = net.mlpfwd(data_validate[:,0:number_predictors+1])




whole_width = 123
whole_height = 123
# img = Image.new( whole_image.mode, whole_image.size)
img = Image.new('RGB', (whole_width, whole_height))
classified_pixels = img.load()
print(img.size)

i = 0
for x in range(whole_width):
    for y in range(whole_height):
        pixel = (prediction[i] * 255).astype(np.int)
        classified_pixels[x, y] = tuple(pixel)
        i = i + 1

img.show()
img.save("classificated_pic.jpg")
