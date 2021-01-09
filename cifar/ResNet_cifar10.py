from functools import reduce

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def myopininon2(x, alpha=1.0):
    return tf.keras.activations.relu(x + alpha) - alpha

def swish(inputs):
    return inputs * tf.math.sigmoid(inputs)

get_custom_objects().update({'Mish': Mish(mish)})


def compose(*funcs):
    """複数の層を結合する。
    """
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def ResNetConv2D(*args, **kwargs):
    """conv を作成する。
    """
    conv_kwargs = {
        "strides": (1, 1),
        "padding": "same",
        "kernel_initializer": "he_normal",
        "kernel_regularizer": l2(1.0e-4),
    }
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)


def bn_act_conv(act, *args, **kwargs):
    """batch mormalization -> ReLU -> conv を作成する。
    """
    return compose(
        BatchNormalization(), Activation(act), ResNetConv2D(*args, **kwargs)
    )


def shortcut(x, residual):
    """shortcut connection を作成する。
    """
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        # x と residual の形状が同じ場合、なにもしない。
        shortcut = x
    else:
        # x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))

        shortcut = Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(stride_w, stride_h),
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1.0e-4),
        )(x)
    return Add()([shortcut, residual])


def basic_block(filters, first_strides, is_first_block_of_first_layer, act):
    """bulding block を作成する。
        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    """

    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_act_conv(
                act, filters=filters, kernel_size=(3, 3), strides=first_strides,
            )(x)

        conv2 = bn_act_conv(act, filters=filters, kernel_size=(3, 3))(conv1)

        return shortcut(x, conv2)

    return f


def bottleneck_block(filters, first_strides, is_first_block_of_first_layer, act):
    """bottleneck bulding block を作成する。
        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    """

    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_act_conv(
               act, filters=filters, kernel_size=(1, 1), strides=first_strides,
            )(x)

        conv2 = bn_act_conv(act, filters=filters, kernel_size=(3, 3))(conv1)
        conv3 = bn_act_conv(act, filters=filters * 4, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3)

    return f


def residual_blocks(block_function, filters, repetitions, is_first_layer, activation):
    """residual block を反復する構造を作成する。
        Arguments:
            block_function: residual block を作成する関数
            filters: フィルター数
            repetitions: residual block を何個繰り返すか。
            is_first_layer: max pooling 直後かどうか
    """

    def f(x):
        for i in range(repetitions):
            # conv3_x, conv4_x, conv5_x の最初の畳み込みは、
            # プーリング目的の畳み込みなので、strides を (2, 2) にする。
            # ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
            # strides を (1, 1) にする。
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(
                filters=filters,
                first_strides=first_strides,
                is_first_block_of_first_layer=(i == 0 and is_first_layer),
                act = activation
            )(x)
        return x

    return f

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

class ResnetBuilder:
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions, activation):
        """ResNet モデルを作成する Factory クラス
        Arguments:
            input_shape: 入力の形状
            num_outputs: ネットワークの出力数
            block_type : residual block の種類 ('basic' or 'bottleneck')
            repetitions: 同じ residual block を何個反復させるか
        """
        # block_type に応じて、residual block を生成する関数を選択する。
        if block_type == "basic":
            block_fn = basic_block
        elif block_type == "bottleneck":
            block_fn = bottleneck_block

        # モデルを作成する。
        ##############################################
        input = Input(shape=input_shape)

        # conv1 (batch normalization -> ReLU -> conv)
        conv1 = compose(
            ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),
            BatchNormalization(),
            Activation(activation),
        )(input)

        # pool
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        # conv2_x, conv3_x, conv4_x, conv5_x
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(
                block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0),
                activation=activation
            )(block)
            filters *= 2

        # batch normalization -> ReLU
        block = compose(BatchNormalization(), Activation(activation))(block)

        # global average pooling
        pool2 = GlobalAveragePooling2D()(block)

        # dense
        fc1 = Dense(
            units=num_outputs, kernel_initializer="he_normal", activation="softmax"
        )(pool2)

        return Model(inputs=input, outputs=fc1)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, activation):
        return ResnetBuilder.build(input_shape, num_outputs, "basic", [2, 2, 2, 2], activation)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, activation):
        return ResnetBuilder.build(input_shape, num_outputs, "basic", [3, 4, 6, 3], activation)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, activation):
        return ResnetBuilder.build(input_shape, num_outputs, "bottleneck", [3, 4, 6, 3], activation)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, activation):
        return ResnetBuilder.build(
            input_shape, num_outputs, "bottleneck", [3, 4, 23, 3], activation
        )

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, activation):
        return ResnetBuilder.build(
            input_shape, num_outputs, "bottleneck", [3, 8, 36, 3], activation
        )



if __name__ == "__main__":

    # mnistのデータ変換
    (x_train_val, y_train_val), (x_test, y_test) = cifar10.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val,
                                                          y_train_val,
                                                          test_size=0.2)

    x_train = preprocess(x_train)
    x_valid = preprocess(x_valid)
    x_test = preprocess(x_test)

    y_train = preprocess(y_train, label=True)
    y_valid = preprocess(y_valid, label=True)
    y_test = preprocess(y_test, label=True)

    INPUT_SHAPE = (32, 32, 3)
    NB_CLASSES = 10
    NB_EPOCH = 50
    BATCH_SIZE = 256
    VERBOSE = 1
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
    momentum = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
    activations = ['Mish']
    dataset = 'cifar10'

    for act in activations:
        print(f"Current Activation funtction:{act}")
        ResNetModel = ResnetBuilder.build_resnet_34(INPUT_SHAPE, NB_CLASSES, act)
        ResNetModel.compile(optimizer=momentum,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    
        history = ResNetModel.fit(x_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=NB_EPOCH,
                            verbose=VERBOSE,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(x_valid, y_valid))
        

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
        np.savetxt(f'./resnet/{dataset}_{act}_50_2.csv', [loss, acc, val_loss, val_acc])
