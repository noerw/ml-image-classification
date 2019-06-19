import sys, os
from PIL import Image
import numpy as np
from marsland_example import mlp

scriptDir = os.path.dirname(__file__)

sky = Image.open(scriptDir + '/trainingdata/sky/trainFalse1.jpg')
river = Image.open(scriptDir + "/trainingdata/river/trainFalse2.jpg")
other_data = Image.open(scriptDir + "/trainingdata/other/train2.jpg")
whole_image = Image.open(scriptDir + "/../data/drone150meter.jpg")
classified_image = Image.open(scriptDir + "/../data/classified_image.jpg")
whole_width, whole_height = whole_image.size

def getAllPixels(image, klasse):
    pixels = image.load()
    width, height = image.size
    all_pixels = []
    for x in range(width):
        for y in range(height):
            px = list(pixels[x, y])
            px.append(klasse)
            all_pixels.append(px)
    return all_pixels

sky_data = getAllPixels(sky, 1)
river_data = getAllPixels(river, 2)
others_data = getAllPixels(other_data, 3)
whole_image_data = getAllPixels(whole_image, 4)
all_training_data = np.array(sky_data + river_data + others_data)

number_predictors = len(all_training_data[0]) - 1

net = mlp.mlp(all_training_data[:,0:number_predictors], all_training_data[:,number_predictors:],1)
net.mlptrain(all_training_data[:,0:number_predictors], all_training_data[:,number_predictors:], 0.25, 100)
print(net)
net.confmat(all_training_data[:,0:number_predictors], all_training_data[:,number_predictors:])
prediction = net.mlpfwd(whole_image_data)
print(len(prediction))
print(prediction[0])

i = 0
img = Image.new( whole_image.mode, whole_image.size)
classified_pixels = img.load()
for x in range(whole_width):
    for y in range(whole_height):
        if int(prediction[i][0]) == 1:
            classified_pixels[x, y] = (255,0,0)
        elif int(prediction[i][0]) == 2:
            classified_pixels[x, y] = (0, 255, 0)
        else:
            classified_pixels[x, y] = (0, 0, 255)
        i = i + 1
img.show()       
img.save("classificated_pic.jpg")