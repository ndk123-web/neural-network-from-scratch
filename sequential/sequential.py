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
        self.loss_history = []

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epoch=10, verbose=True, optimizer=None):
        self.loss_history = []
        X = np.asarray(X)
        y = np.asarray(y)

        # Prefer fit() optimizer when provided; otherwise use compile() value.
        selected_optimizer = (optimizer or getattr(self, "optimizer", "batch")).lower()

        if selected_optimizer == "batch":
            for i in range(epoch):
                # ---------- FORWARD ----------
                output = X
                for layer in self.layers:
                    output = layer.forward(output)

                # ---------- LOSS ----------
                loss = self.loss.loss(y, output)
                self.loss_history.append(float(loss))

                # backward--
                # dL/dA
                dA = self.loss.backward(output, y)

                for layer in reversed(self.layers):

                    if layer.activation_required == True:

                        # dA/dZ
                        dZ = layer.activation_function.backward(dA)

                        # send dA/dZ * dL/dA to current layer, it returns
                        dA = layer.backward(dZ, self.lr)

                    else:
                        dA = layer.backward(dA, self.lr)

                if verbose and i % 100 == 0:
                    print(f"Epoch {i}, Loss: {loss}")

            return self.loss_history

        # SGD
        elif selected_optimizer == "sgd":
            for i in range(epoch):
                # ---------- FORWARD ----------
                epoch_losses = []

                for idx in range(X.shape[0]):
                    each_x = X[idx : idx + 1]
                    each_y = y[idx : idx + 1]

                    output = each_x
                    for layer in self.layers:
                        output = layer.forward(output)

                    # ---------- LOSS ----------
                    loss = self.loss.loss(each_y, output)
                    epoch_losses.append(float(loss))

                    # backward--
                    # dL/dA
                    dA = self.loss.backward(output, each_y)

                    for layer in reversed(self.layers):

                        if layer.activation_required == True:

                            # dA/dZ
                            dZ = layer.activation_function.backward(dA)

                            # send dA/dZ * dL/dA to current layer, it returns
                            dA = layer.backward(dZ, self.lr)

                        else:
                            dA = layer.backward(dA, self.lr)

                epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                self.loss_history.append(epoch_loss)

                if verbose and i % 100 == 0:
                    print(f"Epoch {i}, Loss: {epoch_loss}")

            return self.loss_history

        else:
            raise ValueError(f"Unsupported optimizer: {selected_optimizer}")
