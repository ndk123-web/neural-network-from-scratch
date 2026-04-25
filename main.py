import numpy as np
import pandas as pd
from dense.dense_layer import Dense_Layer

"""
    Testing 
"""


def test1():
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [1, 1, 2],
        ]
    )
    y = [1, 1, 1, 0]
    layer1 = Dense_Layer(3, 4)
    pred1 = layer1.forward(
        X, y, learning_rate=0.01, activation_function="leaky_relu", loss="gradient"
    )

    layer2 = Dense_Layer(4, 3)
    pred2 = layer2.forward(pred1, y, activation_function="relu")
    print(pred2)


def main():
    test1()


if __name__ == "__main__":
    main()
