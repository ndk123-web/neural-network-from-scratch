import numpy as np


class Loss_MSE:

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)
