import numpy as np


class Activation_SoftPlus:
    """
    Activation Function Introduces the Non Linearity in Neural Network

     function forward:
         -  Its responsible for converting weighted sums from previous layer neurons into (0 to +inf) range (thats non linearity)
    """

    def __init__(self):
        pass

    def forward(self, weighted_sum_arr):
        self.Z = weighted_sum_arr
        return np.log1p(np.exp(weighted_sum_arr))

    def backward(self, dA):
        sigmoid_z = 1 / (1 + np.exp(-self.Z))
        return dA * sigmoid_z
