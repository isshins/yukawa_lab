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
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# 名前定義
dataset = "mnist"
act = "relu"


# mnistのデータ変換
(x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
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
        data = data.reshape((-1, 28, 28, 1))

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

    model.add(Conv2D(20, (5, 5), name='conv1', input_shape=(28, 28, 1)))
    model.add(Conv2D(50, (5, 5), name='conv2', input_shape=(24, 24, 1)))
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


# モデルのコンパイル
model = model_sequential(act)
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習処理
batch_size = 128
epochs = 10
verbose = 1
steps_per_epoch = x_train.shape[0] // batch_size,

# 何回試行するか設定している
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    steps_per_epoch=steps_per_epoch[0],
                    validation_data=(x_valid, y_valid))

# 学習済みモデルを保存
model.save(f'{dataset}_{act}.h5')

# 学習経過をグラフで表示
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Val accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Val loss')
plt.legend()
plt.show()

np.savetxt('./{dataset}_{act}.csv', [loss, acc, val_loss, val_acc])

# 評価
#def key_sort_by_num(x):
#    re_list = re.findall(r"[0-9]+", x)
#    re_list = list(map(int, re_list))
#    return re_list
#
#
#def list_from_dir(dir, target_ext=None):
#    data_list = []
#    fnames = os.listdir(dir)
#    fnames = sorted(fnames, key=key_sort_by_num)
#    for fname in fnames:
#        if target_ext is None:
#            path = os.path.join(dir, fname)
#            data_list.append(path)
#        else:
#            _, ext = os.path.splitext(fname)
#            if ext.lower() in target_ext:
#                path = os.path.join(dir, fname)
#                data_list.append(path)
#    return data_list
#
#
#def latest_weight(log_dir):
#    weight_paths = list_from_dir(log_dir, '.hdf5')
#    return weight_paths[-1]
#
#
#model = model_sequential()
#
#ckpt = latest_weight(log_dir)
#model.load_weights(ckpt)
#
#model.compile(optimizer='sgd', loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
#score = model.evaluate(x_test,  y_test)
#print(list(zip(model.metrics_names, score)))
## [('loss', 0.03808286426122068), ('acc', 0.9879)]
#
#plt.figure(figsize=(10, 10))
#
#for i in range(10):
#    data = [(x, t) for x, t in zip(_x_test, _y_test) if t == i]
#    x, y = data[0]
#
#    pred = model.predict(preprocess(x, label=False))
#
#    ans = np.argmax(pred)
#    score = np.max(pred) * 100
#
#    plt.subplot(5, 2, i+1)
#    plt.axis("off")
#    plt.title("ans={} score={}\n{}".format(ans, score, ans == y))
#
#    plt.imshow(x, cmap='gray')
#
#
#plt.tight_layout()
#plt.show()
