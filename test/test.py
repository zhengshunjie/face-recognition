import os
import cv2
import numpy as np

import encoder

imagepath = 'test.jpg'

faceencoder = encoder.FaceEncoder()

image = cv2.imread(imagepath)
image = cv2.resize(image, (160,160))
test_embedding = faceencoder.generate_embedding(image)

print test_embedding
print 'done'

