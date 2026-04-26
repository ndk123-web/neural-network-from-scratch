import numpy as np


class Activation_Tanh:
    """
    Activation Function Introduces the Non Linearity in Neural Network

     function forward:
         -  Its responsible for converting weighted sums from previous layer neurons into (-1 to 1) range (thats non linearity)
    """

    def __init__(self):
        pass

    def forward(self, weighted_sum_arr):
        self.output = (np.exp(weighted_sum_arr) - (np.exp(-weighted_sum_arr))) / (
            np.exp(weighted_sum_arr) + (np.exp(-weighted_sum_arr))
        )
        return self.output

    def backward(self, dA):
        return dA * (1 - np.square(self.output))
