import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import tensorflowjs as tfjs

import cv2
import os
import glob

from PIL import Image

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.layers.merge import concatenate
from skimage import exposure


img_dir_train = "/home/srija/tensorflow/DogCat/dogs-vs-cats/train" # Enter Directory of all images 
img_dir_test = "/home/srija/tensorflow/DogCat/dogs-vs-cats/test_selected" # Enter Directory of all images 
img_dir_test_new = "/home/srija/tensorflow/DogCat/dogs-vs-cats/test1" # Enter Directory of all images 

data_path_train = os.path.join(img_dir_train,'*g')
data_path_test = os.path.join(img_dir_test,'*g')
data_path_test_new = os.path.join(img_dir_test_new,'*g')

num_train = 25000
num_test = 12500

files = glob.glob(data_path_train)
count = 0
data = []
for f1 in sorted(files):
    count = count+1
    #if 1<= count <= 2000:
    image = cv2.imread(f1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = np.array(img)
    img = np.reshape(img, (28*28, 1))
    data.append(img)
    #if 12501<= count <= 14500:
        # image = cv2.imread(f1)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (28,28))
        # img = np.array(img)
        # img = exposure.equalize_hist(img)
        # img = np.reshape(img, (28*28, 1))
        # data.append(img)

train_data = data
train_data = np.array(train_data)
print(train_data.shape)
train_data = np.reshape(train_data, (num_train, 784))
print(train_data.shape)


files = glob.glob(data_path_test_new)
data = []
for f1 in sorted(files):
    image = cv2.imread(f1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = np.array(img)
    img = exposure.equalize_hist(img)
    img = np.reshape(img, (28*28, 1))
    data.append(img)
    
test_data = data
test_data = np.array(test_data)
test_data = np.reshape(test_data, (num_test, 784))
print(test_data.shape)

y = np.random.randint(2, size = (num_train,1))
for i in range(num_train/2):
    y[i] = 0
for i in range(num_train/2,num_train):
    y[i] = 1
#for i in range(20,30):
#    y[i] = 2
#print(y)
print(y.shape)
train_labels = keras.utils.to_categorical(y, num_classes=2)
#print(train_labels)
print(train_labels.shape)



y = np.random.randint(2, size = (num_test,1))
for i in range(num_test/2):
    y[i] = 0
for i in range(num_test/2,num_test):
    y[i] = 1
#for i in range(4,6):
#    y[i] = 2
#print(y)
test_labels = keras.utils.to_categorical(y, num_classes=2)
#print(test_labels)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train_data)))
print("- Test-set:\t{}".format(len(test_data)))
print("- Train-labels:\t\t{}".format(len(train_labels)))
print("- Test-labels:\t\t{}".format(len(test_labels)))



img_size = 28


# The images are stored in one-dimensional arrays of this length.
img_size_flat = img_size*img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

img_shape_full = (img_size, img_size, 1)

# 1 for grayscale
num_channels = 1

# Number of classes, one class for each of 2 classes.
num_classes = 2

input_shape = (img_size*img_size, 1)

initializer = keras.initializers.glorot_normal(seed=None)
# initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)

leakyRelu = keras.layers.LeakyReLU(alpha=0.3)

visible = Input(shape=((img_size)*img_size, 1))
visible1 = Reshape(img_shape_full, input_shape=input_shape)(visible)
#visible1 = BatchNormalization()(visible1)
# first feature extractor
conv1 = Conv2D(64, kernel_size=7, strides = 1,  kernel_initializer=initializer,
                activation=leakyRelu, padding = 'same')(visible1)
pool1 = MaxPooling2D(pool_size=(3, 3), strides = 2)(conv1)
conv1 = Conv2D(64, kernel_size=3, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(pool1)
pool1 = MaxPooling2D(pool_size=(3, 3), strides = 2)(conv1)

conv_1 = Conv2D(64, kernel_size=1, strides = 1, activation=leakyRelu, kernel_initializer=initializer,
                 padding = 'same')(pool1)
conv_1 = Dropout(0.2)(conv_1)
conv_1 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_1)

conv_3 = Conv2D(64, kernel_size=3, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(pool1)
conv_3 = Dropout(0.2)(conv_3)
conv_3 = Conv2D(64, kernel_size=3, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(conv_3)
conv_3 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_3)

conv_5 = Conv2D(64, kernel_size=5, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
               padding = 'same')(pool1)
conv_5 = Dropout(0.2)(conv_5)
conv_5 = Conv2D(64, kernel_size=5, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                  padding = 'same')(conv_5)
conv_5 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_5)

conv_9 = Conv2D(64, kernel_size=9, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                padding = 'same')(pool1)
conv_9 = Dropout(0.2)(conv_9)
conv_9 = Conv2D(64, kernel_size=9, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                  padding = 'same')(conv_9)
conv_9 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_9)

pool_proj_1 = MaxPooling2D(pool_size=(3, 3), strides = 1)(pool1)

pool_proj_1 = Conv2D(64, kernel_size=1, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                  padding = 'same')(pool_proj_1)

merge_1 = concatenate([conv_1, conv_3, conv_5, conv_9, pool_proj_1])

#___________________________2nd time__________________________________#

conv_2_1 = Conv2D(64, kernel_size=1, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(merge_1)
conv_2_1 = Dropout(0.2)(conv_2_1)
conv_2_1 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_2_1)


conv_2_3 = Conv2D(92, kernel_size=3, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                padding = 'same')(merge_1)
conv_2_3 = Dropout(0.2)(conv_2_3)
conv_2_3 = Conv2D(92, kernel_size=3, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(conv_2_3)
conv_2_3 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_2_3)

conv_2_5 = Conv2D(92, kernel_size=5, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                 padding = 'same')(merge_1)
conv_2_5 = Dropout(0.2)(conv_2_5)
conv_2_5 = Conv2D(92, kernel_size=5, strides = 1, activation=leakyRelu,kernel_initializer=initializer,
                  padding = 'same')(conv_2_5)
conv_2_5 = MaxPooling2D(pool_size=(3, 3), strides = 1)(conv_2_5)


merge_3 = concatenate([conv_2_1, conv_2_3, conv_2_5])


flat_0 = Flatten()(merge_3)

output1 = Dense(128, activation=leakyRelu,kernel_initializer=initializer,
                )(flat_0)
output2 = Dense(num_classes, activation='softmax')(output1)

model = Model(inputs=visible, outputs=output2)



print(model.summary())

train_data = np.expand_dims(train_data, 2)
test_data = np.expand_dims(test_data, 2)


from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr = 1e-5)

model.compile(optimizer = optimizer,
                loss = 'categorical_crossentropy',
                metrics = ["accuracy"])

model.fit(x = train_data,
            y = train_labels,
            epochs = 100, batch_size =128)


result = model.evaluate(x = test_data,
                        y = test_labels)



for name,value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2}".format(model.metrics_names[1], result[1]))
    
