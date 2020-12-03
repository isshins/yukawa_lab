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
    myopinion = np.loadtxt('myopinion/mnist_myopinion.csv')
    fts = np.loadtxt('fts/Fts_mnist.csv')
    fts_t = np.loadtxt('fts/Fts_t_mnist.csv')
    fts09 = np.loadtxt('fts/Fts09_mnist.csv')

    data_type = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    print(relu[3][9])
    print(swish[3][9])
    print(fts[3][9])
    print(fts09[3][9])
    print(fts_t[3][9])

    for i in range(4):
        epochs = range(1, len(relu[0])+1)
        plt.plot(epochs, relu[i], 'b', label='relu')
        plt.plot(epochs, swish[i], '#4daf4a', label='swish')
        plt.plot(epochs, fts[i], '#e41a1c', label='fts')
        plt.plot(epochs, fts09[i], '#984ea3', label='fts09')
        plt.plot(epochs, fts_t[i], 'y', label='fts_t')
        plt.xlabel("Number of Epoches")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_epochs()
