import numpy as np


class Loss_BinaryLossEntropy:

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        # very important to prevent from log(0) because if its 0 then its inf and further output breks as Nan
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        return -np.mean((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)))
