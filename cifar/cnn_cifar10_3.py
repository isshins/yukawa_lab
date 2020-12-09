import os
import re
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras.models import Model
from keras import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 名前定義
dataset = "cifar10"
activations = ["relu", "Swish", "Mish", "Tanexp", "softplus", "elu", "selu", "Myopinion1", "Myopinion2"]
activations = ["Swish"]

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'

def swish(inputs, alpha=5.0):
    return inputs * tf.math.sigmoid(inputs+alpha)


class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class Tanexp(Activation):

    def __init__(self, activation, **kwargs):
        super(Tanexp, self).__init__(activation, **kwargs)
        self.__name__ = 'Tanexp'

def tanexp(inputs):
    return inputs * tf.math.tanh(tf.math.exp(inputs))

class Myopinion1(Activation):

    def __init__(self, activation, **kwargs):
        super(Myopinion1, self).__init__(activation, **kwargs)
        self.__name__ = 'Myopinion1'

def myopinion1(inputs):
    return tf.keras.activations.relu(inputs) - 0.1

class Myopinion2(Activation):

    def __init__(self, activation, **kwargs):
        super(Myopinion2, self).__init__(activation, **kwargs)
        self.__name__ = 'Myopinion2'

def myopinion2(inputs, alpha=0.1):
    return tf.keras.activations.relu(inputs + alpha) - alpha

class Fts(Activation):
    def __init__(self, activation, **kwargs):
        super(Fts, self).__init__(activation, **kwargs)
        self.__name__ = 'Fts'

def fts(inputs):
    return inputs * tf.math.sigmoid(tf.keras.activations.relu(inputs))


class Fts_m(Activation):
    def __init__(self, activation, **kwargs):
        super(Fts_m, self).__init__(activation, **kwargs)
        self.__name__ = 'Fts_m'

def fts_m(inputs):
    return swish(myopininon2(inputs, 1.27846454))

                                
get_custom_objects().update({'Swish': Swish(swish)})
get_custom_objects().update({'Mish': Mish(mish)})
get_custom_objects().update({'Tanexp': Tanexp(tanexp)})
get_custom_objects().update({'Myopinion1': Myopinion1(myopinion1)}) 
get_custom_objects().update({'Myopinion2': Myopinion2(myopinion2)}) 
get_custom_objects().update({'Fts': Fts(fts)})
get_custom_objects().update({'Fts_m': Fts_m(fts_m)})

# mnistのデータ変換
(x_train_val, y_train_val), (x_test, y_test) = cifar10.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2)

# データの前処理
def preprocess(data, label=False):
    if label:
        # 教師データはto_categorical()でone-hot-encodingする。
        data = to_categorical(data)
    else:
        # 入力画像は、astype('float32')で型変換を行い、レンジを0-1にするために255で割る。
        # 0-255 -> 0-1
        data = data.astype('float32') / 255
        # Kerasの入力データの形式は(ミニバッチサイズ、横幅、縦幅、チャネル数)である必要があるので、reshape()を使って形式を変換する。
        # (sample, width, height) -> (sample, width, height, channel)
        data = data.reshape((-1, 32, 32, 3))

    return data


x_train = preprocess(x_train)
x_valid = preprocess(x_valid)
x_test = preprocess(x_test)

y_train = preprocess(y_train, label=True)
y_valid = preprocess(y_valid, label=True)
y_test = preprocess(y_test, label=True)

# モデル作成
def model_sequential(activation):
    model = models.Sequential()

    model.add(Conv2D(20, (5, 5), name='conv1', input_shape=(32, 32, 3)))
    model.add(Conv2D(50, (5, 5), name='conv2', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(500, name='dense1'))

    model_add_block(model, 12, activation)

    model.add(Dense(10, name='dense2'))
    model.add(Activation('softmax', name='last_act'))

    return model


def model_add_block(model, layers, activation):
    for times in range(layers):
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(0.25))

    return model

# 学習処理
batch_size = 128
epochs = 50
verbose = 1
steps_per_epoch = x_train.shape[0] // batch_size

# モデルのコンパイル
for act in activations:
    
    model = model_sequential(act)
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 何回試行するか設定している
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(x_valid, y_valid))
    
    # 学習経過をグラフで表示
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    np.savetxt(f'./cnn/{dataset}_{act}_50.csv', [loss, acc, val_loss, val_acc])
