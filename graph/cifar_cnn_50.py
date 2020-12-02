import numpy as np
import matplotlib.pyplot as plt


def conmprison_cifar():
    relu = np.loadtxt('cifar/cnn/cifar10_relu.csv')
    selu = np.loadtxt('cifar/cnn/cifar10_selu.csv')
    swish = np.loadtxt('cifar/cnn/cifar10_Swish.csv')
    mish = np.loadtxt('cifar/cnn/cifar10_Mish.csv')
    tanexp = np.loadtxt('cifar/cnn/cifar10_Tanexp.csv')
    softplus = np.loadtxt('cifar/cnn/cifar10_softplus.csv')
    elu = np.loadtxt('cifar/cnn/cifar10_elu.csv')
    myopinion1 = np.loadtxt('cifar/cnn/cifar10_Myopinion1.csv')
    myopinion2 = np.loadtxt('cifar/cnn/cifar10_Myopinion2.csv')

    data_type = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    print(relu[3][9])
    print(selu[3][9])
    print(swish[3][9])
    print(mish[3][9])
    print(tanexp[3][9])
    print(softplus[3][9])
    print(elu[3][9])
    print(myopinion1[3][9])
    print(myopinion2[3][9])

    for i in range(4):
        epochs = range(1, len(relu[0])+1)
        plt.plot(epochs, relu[i], 'b', label='relu')
        plt.plot(epochs, selu[i], 'r', label='selu')
        plt.plot(epochs, swish[i], '#4daf4a', label='swish')
        plt.plot(epochs, mish[i], 'g', label='mish')
        plt.plot(epochs, tanexp[i], 'y', label='tanexp')
        plt.plot(epochs, softplus[i], 'c', label='softplus')
        plt.plot(epochs, elu[i], 'm', label='elu')
        plt.plot(epochs, myopinion1[i], 'k', label='myopinion1')
        plt.plot(epochs, myopinion2[i], '#984ea3', label='myopinion2')
        plt.xlabel("Number of Epoches")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_cifar()
