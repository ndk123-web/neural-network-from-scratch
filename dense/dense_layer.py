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
        y,
        learning_rate=0.01,
        activation_function="relu",
        loss="categorical",
        optimizer="gradient",
        epoch=1000,
        last_layer=False,
    ):
        weighted_sums = np.dot(X, self.weights) + self.biases

        self.activation_function = activation_function
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.last_layer = last_layer

        # Accept either enum values or plain strings from callers.
        if isinstance(activation_function, str):
            activation_function = activation_function.lower()
        elif isinstance(activation_function, Activations):
            activation_function = activation_function.value

        match activation_function:
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

        if self.last_layer:
            match self.loss:
                case "categorical_cross_entropy":
                    self.loss_fn = Loss_CategoricalCrossEntropy()
                case "binary_cross_entropy":
                    self.loss_fn = Loss_BinaryLossEntropy()
                case _:
                    self.loss_fn = Loss_CategoricalCrossEntropy()

            print("Loss: ", self.loss_fn.loss(y, y_pred))
        return y_pred
