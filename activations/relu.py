import numpy as np


class Activation_ReLU:
    """
    Activation Function Introduces the Non Linearity in Neural Network

    function forward:
        -  Its responsible for converting weighted sums from previous layer neurons into (0 to +inf) range (thats non linearity)
    """

    def __init__(self):
        pass

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ
