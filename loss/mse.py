import numpy as np


class Loss_MSE:

    def __init__(self):
        pass

    """
        calculates actual cost 
    """

    def loss(self, y, y_pred):
        y_pred = np.array(y_pred)
        y = np.array(y)
        if y.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y = y.reshape(-1, 1)
        return np.mean((y_pred - y) ** 2)

    """
        returns dA/dZ
    """

    def backward(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        if y_true.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_true = y_true.reshape(-1, 1)
        return (2 * (y_pred - y_true)) / y_pred.shape[0]
