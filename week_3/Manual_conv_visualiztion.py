import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models
import cv2
from scipy import misc

# generate an image using misc
i = misc.ascent()

# visualizing the image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()
plt.imsave(r'C:\TF_professional_training\TF-Developer-Professional-Certificate\week_3\original',i)


# split x and y of image
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]


# create a 3x3 filter sharp edges
filter = [[0,1,0],[1,-4,1],[0,1,0]]

# vertical lines filter
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# horizantal lines filter
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


# max pooling

weight = 1

# creating a convultion

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x - 1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[0][1])
      convolution = convolution + (i[x + 1, y-1] * filter[0][2])
      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution


# Ploting the image.
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()

plt.imsave(r'C:\TF_professional_training\TF-Developer-Professional-Certificate\week_3\conv',i_transformed)

# create a 2x2 max pool and ploting it
new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x + 1, y])
        pixels.append(i_transformed[x, y + 1])
        pixels.append(i_transformed[x + 1, y + 1])
        newImage[int(x / 2), int(y / 2)] = max(pixels)


# the max pool reduce the size by 50% from 512 to  256 pixels
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()
plt.imsave(r'C:\TF_professional_training\TF-Developer-Professional-Certificate\week_3\maxpool',newImage)







