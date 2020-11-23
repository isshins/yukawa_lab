import numpy as np
import matplotlib.pyplot as plt


def conmprison_layers():
    relu = np.loadtxt('relu/layers/mnist_relu_layers.csv')
    selu = np.loadtxt('selu/layers/mnist_selu_layers.csv')
    mish = np.loadtxt('mish/layers/mnist_mish_layers.csv')
    tanexp = np.loadtxt('Tanexp/layers/mnist_tanexp_layers.csv')
    softplus = np.loadtxt('softplus/layers/mnist_softplus_layers.csv')
    elu = np.loadtxt('elu/layers/mnist_elu_layers.csv')
    myopinion = np.loadtxt('myopinion/layers/mnist_myopinion_layers.csv')

    data_type = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    layers = range(12, 23)
    print(relu[3][9])
    print(selu[3][9])
    print(mish[3][9])
    print(tanexp[3][9])
    print(softplus[3][9])
    print(elu[3][9])
    print(myopinion[3][9])
    for i in range(4):
        plt.plot(layers, relu[i], 'b', label='relu')
        plt.plot(layers, selu[i], 'r', label='selu')
        plt.plot(layers, mish[i], 'g', label='mish')
        plt.plot(layers, tanexp[i], 'y', label='tanexp')
        plt.plot(layers, softplus[i], 'c', label='softplus')
        plt.plot(layers, elu[i], 'm', label='elu')
        plt.plot(layers, myopinion[i], 'k', label='myopinion')
        plt.xlabel("Number of layers")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_layers()
