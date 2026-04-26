import numpy as np


class Activation_Sigmoid:
    """
    Activation Function Introduces the Non Linearity in Neural Network

    function forward:
        -  Its responsible for converting weighted sums from previous layer neurons into (0-1) range (thats non linearity)
    """

    def __init__(self):
        pass

    def forward(self, weighted_sums_arr):
        self.output = 1 / (1 + np.exp(-weighted_sums_arr))
        return self.output

    def backward(self, dA):
        return dA * self.output * (1 - self.output)
