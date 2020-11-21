import os
import re
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras.models import Model
from keras import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from mnist_relu_keras import model_sequential, preprocess

(x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2)

# 評価
model = model_sequential()
model.load_weights("./relu.h5")

model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

score = model.evaluate(x_test,  y_test)
print(list(zip(model.metrics_names, score)))

plt.figure(figsize=(10, 10))

for i in range(10):
    data = [(x, t) for x, t in zip(_x_test, _y_test) if t == i]
    x, y = data[0]

    pred = model.predict(preprocess(x, label=False))

    ans = np.argmax(pred)
    score = np.max(pred) * 100

    plt.subplot(5, 2, i+1)
    plt.axis("off")
    plt.title("ans={} score={}\n{}".format(ans, score, ans == y))

    plt.imshow(x, cmap='gray')


plt.tight_layout()
plt.show()
