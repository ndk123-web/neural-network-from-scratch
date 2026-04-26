import numpy as np
from activations.relu import Activation_ReLU
from activations.sigmoid import Activation_Sigmoid
from activations.leaky_relu import Activation_LeakyReLU
from activations.tanh import Activation_Tanh
from loss.binary_cross_entropy import Loss_BinaryLossEntropy
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from enum import Enum


class Activations(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKYRELU = "leaky_relu"


class Dense_Layer:
    """
    NOTE:
        -  My Idea is User will create Object of Dense_Layer (here one object means 1 layer of neurons)
        - layer1 = Dense_Layer(4,5)
        - NOTE:
            -  Here 4 is current layer neurons, 5 is next layer neurons this should be known to user
        - function forward:
            - it will just forward the predictions to the next layers of neurons

    NOTE:
        - Each Neurons has (len(n_inputs[0]) weights) and (1 bias)

    TODO:
        - Feed Forward
        - Back Propagation (includes loss and calculas)
    """

    def __init__(self, n_inputs, n_neurons):
        self.name = "Ndk"
        self.weights = 0.5 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0.02 * np.random.randn(1, n_neurons)
        self.activation_function = "relu"
        self.epoch = 1000

    # i will do this after some time
    def validate(self):
        pass

    def forward(
        self,
        X,
        learning_rate=0.01,
        activation_function="relu",
        loss="categorical",
        optimizer="gradient",
        epoch=1000,
    ):
        self.X = X
        weighted_sums = np.dot(X, self.weights) + self.biases

        self.activation_name = activation_function
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch

        # Accept either enum values or plain strings from callers.
        if isinstance(activation_function, str):
            activation_function = activation_function.lower()
        elif isinstance(activation_function, Activations):
            activation_function = activation_function.value

        match self.activation_name:
            case Activations.SIGMOID:
                self.activation_function = Activation_Sigmoid()
            case Activations.RELU:
                self.activation_function = Activation_ReLU()
            case Activations.TANH:
                self.activation_function = Activation_Tanh()
            case Activations.LEAKYRELU:
                self.activation_function = Activation_LeakyReLU()
            case _:
                self.activation_function = Activation_ReLU()

        y_pred = self.activation_function.forward(weighted_sums)
        self.output = y_pred
        return y_pred

    """
    TODO:
        1. get the loss by previos
    """

    def backward(self, dZ, lr):
        m = self.X.shape[0]

        dW = (1 / m) * self.X.T @ dZ
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.weights.T

        self.weights -= lr * dW
        self.biases -= lr * db

        return dA_prev
