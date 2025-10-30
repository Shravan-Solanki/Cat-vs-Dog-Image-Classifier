import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

img = cv2.imread("train/dog.9.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap="gray")
plt.show()

img_size = 50

new_array = cv2.resize(img,(img_size,img_size))
plt.imshow(new_array,cmap="gray")
plt.show()

training_data = []
# cat = 0 && dog = 1
def create_training_data():
  path = "train"
  for img in os.listdir(path):
    try:
      if('cat' in img):
        cl = 0
      else:
        cl = 1
      img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
      new_array = cv2.resize(img_array,(img_size,img_size))
      training_data.append([new_array,cl])
    except Exception as e:
      pass

create_training_data()

import random

random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1,img_size,img_size,1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import time

X = X / 255.0

dense_layer = 0
layer_size = 64
conv_layer = 3

model = Sequential()
model.add(Conv2D(layer_size,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

for l in range(conv_layer-1):
  model.add(Conv2D(layer_size,(3,3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

for l in range(dense_layer):
  model.add(Dense(layer_size))
  model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

y = np.array(y)
model.fit(X,y,batch_size=32,validation_split=0.3,epochs=10)

model.save('64x3-CNN.keras')

categories = ["cat","dog"]

def prepare(filepath):
  img_size = 50
  img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array,(img_size,img_size))
  return new_array.reshape(-1,img_size,img_size,1)

model = tf.keras.models.load_model("64x3-CNN.keras")
p = prepare("test1/2017.jpg")
prediction = model.predict([p])
print(categories[int(prediction[0][0])])

p = cv2.imread("test1/2017.jpg")
plt.imshow(p)
plt.show()

