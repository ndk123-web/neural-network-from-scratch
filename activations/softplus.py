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
        return np.log(1 + np.exp(weighted_sum_arr))
