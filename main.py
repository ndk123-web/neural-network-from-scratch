import numpy as np
import pandas as pd
from activations.softmax import Activation_SoftMax
from dense.dense_layer import Dense_Layer
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
    model.add(Dense_Layer(4, 2, activation_fn="softmax"))

    model.compile(loss=Loss_CategoricalCrossEntropy(), lr=0.001)

    model.fit(X, y, epoch=1000)


def main():
    test1()


if __name__ == "__main__":
    main()
