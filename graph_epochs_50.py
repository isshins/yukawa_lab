import numpy as np
import matplotlib.pyplot as plt


def conmprison_epochs_50():
    relu = np.loadtxt('relu/mnist_relu_50.csv')
    selu = np.loadtxt('selu/mnist_selu_50.csv')
    swish = np.loadtxt('swish/mnist_Swish_50.csv')
    mish = np.loadtxt('mish/mnist_mish_50.csv')
    tanexp = np.loadtxt('Tanexp/mnist_tanexp_50.csv')
    softplus = np.loadtxt('softplus/mnist_softplus_50.csv')
    elu = np.loadtxt('elu/mnist_elu_50.csv')
    myopinion = np.loadtxt('myopinion/mnist_myopinion_50.csv')

    data_type = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    print(relu[3][9])
    print(selu[3][9])
    print(swish[3][9])
    print(mish[3][9])
    print(tanexp[3][9])
    print(softplus[3][9])
    print(elu[3][9])
    print(myopinion[3][9])

    for i in range(4):
        epochs = range(1, len(relu[0])+1)
        plt.plot(epochs, relu[i], 'b', label='relu')
        plt.plot(epochs, selu[i], 'r', label='selu')
        plt.plot(epochs, swish[i], '#4daf4a', label='swish')
        plt.plot(epochs, mish[i], 'g', label='mish')
        plt.plot(epochs, tanexp[i], 'y', label='tanexp')
        plt.plot(epochs, softplus[i], 'c', label='softplus')
        plt.plot(epochs, elu[i], 'm', label='elu')
        plt.plot(epochs, myopinion[i], 'k', label='myopinion')
        plt.xlabel("Number of Epoches")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_epochs_50()
