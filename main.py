import numpy as np
from dense.dense_layer import Dense_Layer
from loss.binary_cross_entropy import Loss_BinaryLossEntropy
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from sequential.sequential import Sequential
from loss.mse import Loss_MSE

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
    X = np.array([[1, 1, 2], [2, 1, 1], [1, 5, 10], [11, 10, 100]])

    y = np.array([9, 10, 20, 50])
    model = Sequential()

    model.add(Dense_Layer(3, 4, activation_fn="relu"))
    model.add(
        Dense_Layer(4, 1, activation_required=False)
    )  # we dont require activation in last

    model.compile(loss=Loss_MSE(), lr=0.0005)

    model.fit(X, y, 2000)


def main():
    test1()  # binary + sigmoid
    test2()  # categorical + softmax
    test3()  # mse + no activation


if __name__ == "__main__":
    main()
