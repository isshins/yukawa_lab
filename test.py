import numpy as np
import matplotlib.pyplot as plt


def conmprison_epochs():
    relu = np.loadtxt('relu/mnist_relu.csv')
    selu = np.loadtxt('selu/mnist_selu.csv')
    swish = np.loadtxt('swish/mnist_Swish.csv')
    mish = np.loadtxt('mish/mnist_mish.csv')
    tanexp = np.loadtxt('Tanexp/mnist_tanexp.csv')
    softplus = np.loadtxt('softplus/mnist_softplus.csv')
    elu = np.loadtxt('elu/mnist_elu.csv')
    myopinion1 = np.loadtxt('myopinion/mnist_myopinion.csv')
    myopinion2 = np.loadtxt('myopinion/mnist_Myopinion2.csv')
    myopinion3 = np.loadtxt('myopinion/mnist_Myopinion3.csv')
    myopinion4 = np.loadtxt('myopinion/mnist_Myopinion4.csv')

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
    print(myopinion3[3][9])
    print(myopinion4[3][9])

    for i in range(4):
        epochs = range(1, len(relu[0])+1)
        plt.plot(epochs, swish[i], 'c', label='swish')
        plt.plot(epochs, myopinion3[i], 'k', label='myopinion1')
        plt.plot(epochs, myopinion4[i], 'g', label='myopinion2')

        plt.xlabel("Number of Epoches")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_epochs()
