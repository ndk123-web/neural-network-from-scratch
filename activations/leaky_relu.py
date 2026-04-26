import numpy as np


class Activation_LeakyReLU:
    """
    Activation Function Introduces the Non Linearity in Neural Network

    function forward:
        -  Its responsible for converting weighted sums from previous layer neurons into (0.1 * +inf to +inf) range (thats non linearity)
    """

    def __init__(self):
        pass

    def forward(self, weighted_sum_arr):
        self.Z = weighted_sum_arr
        # max(0.1*x, x)
        return np.maximum(0.1 * weighted_sum_arr, weighted_sum_arr)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z < 0] *= 0.1
        return dZ
