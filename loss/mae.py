import numpy as np


class Loss_MAE:

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return np.mean((y_pred - y))
