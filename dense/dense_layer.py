import numpy as np
from activations.relu import Activation_ReLU
from activations.sigmoid import Activation_Sigmoid
from activations.leaky_relu import Activation_LeakyReLU
from activations.softplus import Activation_SoftPlus
from activations.softmax import Activation_SoftMax
from activations.tanh import Activation_Tanh
from loss.binary_cross_entropy import Loss_BinaryLossEntropy
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from enum import Enum


class Activations(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKYRELU = "leaky_relu"
    SOFTPLUS = "softplus"
    SOFTMAX = "softmax"


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

    NOTE:
        - In Entire Network 2 main thing is must
            1. Update the weights of each neurons of the layers
            2. current layer must return the self error to the previous layers

    TODO:
        - Feed Forward
        - Back Propagation (includes loss and calculas)
    """

    def __init__(
        self, n_inputs, n_neurons, activation_fn="relu", activation_required=True
    ):
        self.name = "Ndk"
        # Xavier-style scaling gives more stable gradients than large random weights.
        weight_scale = np.sqrt(1.0 / max(1, n_inputs))
        self.weights = np.random.randn(n_inputs, n_neurons) * weight_scale
        self.biases = np.zeros((1, n_neurons))
        self.activation_name = activation_fn
        self.epoch = 1000
        self.activation_required = activation_required

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

        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch

        activation_name = self.activation_name
        if isinstance(activation_name, Activations):
            activation_name = activation_name.value
        activation_name = str(activation_name).lower()

        match activation_name:
            case "sigmoid":
                self.activation_function = Activation_Sigmoid()
            case "relu":
                self.activation_function = Activation_ReLU()
            case "tanh":
                self.activation_function = Activation_Tanh()
            case "leaky_relu":
                self.activation_function = Activation_LeakyReLU()
            case "softplus":
                self.activation_function = Activation_SoftPlus()
            case "softmax":
                self.activation_function = Activation_SoftMax()
            case _:
                self.activation_function = Activation_ReLU()

        # if it not required activations (in case of regression``)
        if self.activation_required == False:
            self.output = weighted_sums
            return self.output

        y_pred = self.activation_function.forward(weighted_sums)
        self.output = y_pred
        return y_pred

    # dZ = dL/dA * dA/dZ
    def backward(self, dZ, lr):
        m = self.X.shape[0]

        # dW = dL/dA * dA/dZ * dZ/dW
        # db = dL/dA * dA/dZ * dZ/db
        dW = (1 / m) * self.X.T @ dZ  # get the slope(gradient)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)  # get the bias

        # we can say dA_prev is current layer error that we need to send to the previous layer
        # it calculates error of current layer to send to the prev layer
        dA_prev = dZ @ self.weights.T

        # updated weights
        self.weights -= lr * dW
        self.biases -= lr * db

        return dA_prev
