import numpy as np


class Activation_SoftMax:

    def __init__(self):
        pass

    def forward(self, weighted_sum_arr):

        total_sum = np.sum(np.exp(weighted_sum_arr))

        pred = [(np.exp(w)) / total_sum for w in weighted_sum_arr]
        return pred
