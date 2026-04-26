import numpy as np


class Activation_SoftMax:

    def __init__(self):
        pass

    def forward(self, weighted_sum_arr):
        weighted_sum_arr = np.array(weighted_sum_arr)
        shifted = weighted_sum_arr - np.max(weighted_sum_arr, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        pred = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return pred

    # softmax actuaaly have jacobian derivative which gives syntactical issue
    def backward(self):
        pass
