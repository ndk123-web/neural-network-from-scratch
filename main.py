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
    pred = layer1.forward(
        X, y, learning_rate=0.01, activation_function="leaky_relu", loss="gradient"
    )
    print(pred)


def main():
    test1()


if __name__ == "__main__":
    main()
