import numpy as np


class Loss_CategoricalCrossEntropy:
    """
    It will use with SoftMax activation function
    """

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        # very important to prevent from log(0) because if its 0 then its inf and further output breks as Nan
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y = np.array(y)

        # Supports both sparse labels (shape: [batch]) and one-hot labels (shape: [batch, classes]).
        if y.ndim == 1:
            correct_confidence = y_pred[np.arange(len(y)), y]
        else:
            correct_confidence = np.sum(y_pred * y, axis=1)

        return -np.mean(np.log(correct_confidence))
