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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.models import load_model


img_dir_train = "/home/srija/tensorflow/DogCat/dogs-vs-cats/train" # Enter Directory of all images 
img_dir_test = "/home/srija/tensorflow/DogCat/dogs-vs-cats/test_selected" # Enter Directory of all images 
img_dir_test_new = "/home/srija/tensorflow/DogCat/dogs-vs-cats/test1" # Enter Directory of all images 

data_path_train = os.path.join(img_dir_train,'*g')
data_path_test = os.path.join(img_dir_test,'*g')
data_path_test_new = os.path.join(img_dir_test_new,'*g')

num_train = 4000
num_test = 12500

files = glob.glob(data_path_train)
count = 0
data = []
for f1 in sorted(files):
    count = count+1
    if 1<= count <= 2000:
        image = cv2.imread(f1)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28))
        #img = np.array(img)
        img = np.reshape(img, (28*28, 1))
        data.append(img)
    if 12501<= count <= 14500:
        image = cv2.imread(f1)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28))
        #img = np.array(img)
        img = np.reshape(img, (28*28, 1))
        data.append(img)

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
    #img = np.array(img)
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

# files = glob.glob(data_path_test_new)
# data = []
# for f1 in files:
#     image = cv2.imread(f1)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (28,28))
#     #img = np.array(img)
#     img = np.reshape(img, (28*28, 1))
#     data.append(img)
# test_data_new = data
# test_data_new = np.array(test_data_new)
# test_data_new = np.reshape(test_data_new, (12500, 784))
# print(test_data_new.shape)





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

# Number of classes, one class for each of 10 digits.
num_classes = 2

model = Sequential()

#initializer = keras.initializers.glorot_normal(seed=None)
initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)

leakyRelu = keras.layers.LeakyReLU(alpha=0.3)

model.add(InputLayer(input_shape = (img_size_flat, )))

model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 16, padding = "same",
            kernel_initializer=initializer,activation = leakyRelu, name = "conv_layer_1"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 36, padding = "same",
            kernel_initializer=initializer,activation = leakyRelu, name = "conv_layer_2"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 56, padding = "same",
            kernel_initializer=initializer,activation = leakyRelu, name = "conv_layer_3"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Conv2D(kernel_size = 5, strides = 1, filters = 80, padding = "same",
            kernel_initializer=initializer,activation = leakyRelu, name = "conv_layer_4"))

model.add(MaxPooling2D(pool_size = 2, strides = 2))

model.add(Flatten())

model.add(Dense(128, kernel_initializer=initializer,activation = leakyRelu))

model.add(Dense(num_classes, activation = "softmax"))

print(model.summary())


from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr = 1e-5)

model.compile(optimizer = optimizer,
                loss = 'categorical_crossentropy',
                metrics = ["accuracy"])

model.fit(x = train_data,
            y = train_labels,
            epochs = 110, batch_size =128)


result = model.evaluate(x = test_data,
                        y = test_labels)



for name,value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2}".format(model.metrics_names[1], result[1]))
    
# y_pred = model.predict(test_data_new)

# print(y_pred)

# cls_pred = np.argmax(y_pred, axis = 1)

# print(cls_pred)

# np.savetxt("output_400.csv", cls_pred)


path_model = 'model.keras'

tfjs.converters.save_keras_model(model, '/home/srija/tensorflow/DogCat/dogs-vs-cats/tfjs/public/models/myModel')
model.save(path_model)

# model2 = load_model(path_model)

images = test_data[0:10]
cls_true = test_labels[0:10]

y_pred_1 = model.predict(x=images)
cls_pred = np.argmax(y_pred_1, axis = 1)

print(cls_true)
print(cls_pred)

print(model.summary())
