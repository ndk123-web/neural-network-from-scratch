import numpy as np
import pandas as pd
from activations.softmax import Activation_SoftMax
from dense.dense_layer import Dense_Layer
from loss.binary_cross_entropy import Loss_BinaryLossEntropy
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from activations.relu import Activation_ReLU
from activations.sigmoid import Activation_Sigmoid
from sequential.sequential import Sequential

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

    model = Sequential()
    model.add(Dense_Layer(2, 4, activation_fn="relu"))
    model.add(Dense_Layer(4, 1, activation_fn="sigmoid"))

    model.compile(loss=Loss_BinaryLossEntropy(), lr=0.001)

    model.fit(X, y, epoch=1000)


def test2():
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
    )
    y = np.array([0, 0, 1, 2])
    model = Sequential()

    model.add(Dense_Layer(3, 3, activation_fn="softplus"))
    model.add(Dense_Layer(3, 3, activation_fn="softmax"))

    model.compile(loss=Loss_CategoricalCrossEntropy(), lr=0.1)

    model.fit(X, y, 2000)


def test3():
    X = []
    model = Sequential()


def main():
    test1()
    # test2()
    # test3()


if __name__ == "__main__":
    main()
