import numpy as np
import pandas as pd
from activations.softmax import Activation_SoftMax
from dense.dense_layer import Dense_Layer
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from activations.relu import Activation_ReLU
from activations.sigmoid import Activation_Sigmoid

"""
    Testing 
"""


def test1():
    X = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
        ]
    )
    y = np.array([0, 1, 0, 1])

    layer1 = Dense_Layer(2, 4)
    activation1 = Activation_ReLU()

    layer2 = Dense_Layer(4, 2)
    activation2 = Activation_SoftMax()

    epoch = 1000
    for i in range(epoch):
        # forward
        Z1 = layer1.forward(X)
        A1 = activation1.forward(Z1)

        Z2 = layer2.forward(A1)
        A2 = activation2.forward(Z2)

        # calculate loss wrt Z
        loss = Loss_CategoricalCrossEntropy()
        dL_by_dZ = loss.backward(A2, y)
        if i % 100 == 0:
            print("Loss:", loss.loss(y, A2))

        # Dense Backward
        dA1 = layer2.backward(dL_by_dZ, 0.01)
        layer1.backward(dA1, 0.1)


def main():
    test1()


if __name__ == "__main__":
    main()
