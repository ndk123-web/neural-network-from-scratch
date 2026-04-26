import numpy as np
from dense.dense_layer import Dense_Layer
from typing import List


class Sequential:
    """
    NOTE:
        - Sequential is One Entire Neural Network
        - Entire Network has 1 loss function
        - Each layer has 1 activation functions
    """

    def __init__(self):
        self.layers: List[Dense_Layer] = []

    def add(self, dense_layer):
        self.layers.append(dense_layer)

    def compile(
        self,
        loss,
        lr,
        optimizer="gradient",
    ):
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer

    def fit(self, X, y, epoch=10):

        # each epoch (Batch Gradient Descent)
        for i in range(epoch):
            # forward
            output = X
            for dense_layer in self.layers:
                output = dense_layer.forward(output)

            # loss
            loss = self.loss.loss(y, output)

            # backward
            dZ = self.loss.backward(output, y)
            for layer in reversed(self.layers):
                dZ = layer.backward(dZ, self.lr)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")
