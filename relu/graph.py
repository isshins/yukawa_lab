import numpy as np
import matplotlib.pyplot as plt


def conmprison_layers():
    relu = np.loadtxt('mnist_relu_layers.csv')
    data_type = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    print(relu[0])
    for i in range(4):
        layers = range(15, 26)
        plt.plot(layers, relu[i], 'b', label='relu')
        plt.xlabel("Number of layers")
        plt.ylabel(f"{data_type[i]}")
        plt.gca().yaxis.set_tick_params(direction='in')
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.legend()
        plt.show()


conmprison_layers()
