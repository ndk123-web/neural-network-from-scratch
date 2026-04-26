# ndk-nn

`ndk-nn` is a NumPy-based neural network implementation that exposes the internal mechanics of forward propagation, backpropagation, and parameter updates.  
The codebase is structured as a minimal training engine with explicit layer, activation, and loss responsibilities.

## Overview

- Implements a sequential feedforward network with dense layers.
- Provides explicit gradient computation and in-place parameter updates.
- Exists to provide a direct, inspectable implementation of training logic commonly abstracted by high-level frameworks.

## Core Features

- Dense layer forward pass: `Z = XW + b`
- Configurable activations per layer (`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `SoftPlus`, `SoftMax`)
- Multiple loss functions (`Binary Cross Entropy`, `Categorical Cross Entropy`, `MSE`, `MAE`)
- Reverse-order backpropagation across stacked layers
- Batch-wise gradient descent weight and bias updates
- Sparse and one-hot label handling for categorical cross-entropy

## Architecture / Components

### `dense/dense_layer.py`

- Defines `Dense_Layer`.
- Owns trainable parameters: `weights`, `biases`.
- Performs linear transform and activation dispatch in `forward`.
- Computes `dW`, `db`, and upstream gradient in `backward`.

### `sequential/sequential.py`

- Defines `Sequential`.
- Stores ordered layers via `add`.
- Configures optimization settings via `compile`.
- Executes training loop via `fit`:
  - layer-wise forward pass
  - loss computation
  - reverse backpropagation

### `activations/*`

- One class per activation function.
- Each module provides:
  - `forward(...)` for activation output
  - `backward(...)` for local derivative application

### `loss/*`

- One class per objective function.
- Each module provides:
  - `loss(y, y_pred)` for scalar objective
  - `backward(y_pred, y_true)` for output gradient

### `main.py`

- Contains runnable training scenarios for binary classification, multi-class classification, and regression.

## Training / Execution Flow

1. Initialize model and append `Dense_Layer` instances.
2. Call `compile(loss=..., lr=...)` to set objective and learning rate.
3. For each epoch in `fit`:
   - propagate input through all layers
   - compute scalar loss from final output
   - compute output gradient from loss
   - backpropagate gradients in reverse layer order
   - update parameters at each dense layer

```text
X -> Dense/Activation -> ... -> Dense/Activation -> y_pred
                                 |
                                 v
                               Loss
                                 |
                                 v
                       Backward gradients (reverse)
```

## Mathematical Foundation

```text
Forward:
Z[l] = A[l-1]W[l] + b[l]
A[l] = f(Z[l])

Loss gradients:
dW[l] = (1/m) * A[l-1]^T dZ[l]
db[l] = (1/m) * sum(dZ[l], axis=0)
dA[l-1] = dZ[l]W[l]^T

Parameter update:
W[l] <- W[l] - lr * dW[l]
b[l] <- b[l] - lr * db[l]
```

## Usage Example

```python
import numpy as np
from dense.dense_layer import Dense_Layer
from sequential.sequential import Sequential
from loss.binary_cross_entropy import Loss_BinaryLossEntropy

X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
y = np.array([0, 1, 0, 1])

model = Sequential()
model.add(Dense_Layer(2, 4, activation_fn="relu"))
model.add(Dense_Layer(4, 1, activation_fn="sigmoid"))
model.compile(loss=Loss_BinaryLossEntropy(), lr=0.001)
model.fit(X, y, epoch=1000)
```

## Project Structure

```text
ndk-nn/
├── activations/
│   ├── relu.py
│   ├── sigmoid.py
│   ├── tanh.py
│   ├── leaky_relu.py
│   ├── softplus.py
│   └── softmax.py
├── dense/
│   └── dense_layer.py
├── loss/
│   ├── binary_cross_entropy.py
│   ├── categorical_cross_entropy.py
│   ├── mse.py
│   └── mae.py
├── sequential/
│   └── sequential.py
└── main.py
```

## Limitations

- Full-batch updates only; no mini-batch data loader.
- No optimizer abstraction beyond direct gradient descent.
- No regularization components (L1/L2, dropout).
- Limited runtime shape/type validation.
- No dedicated inference and evaluation interface.
- No automated unit or integration test suite.

## Roadmap / Future Improvements

- Add mini-batch training support.
- Introduce optimizer classes (`SGD` with momentum, `Adam`).
- Add initialization strategies (`He`, `Xavier`).
- Add explicit `predict` and `evaluate` APIs.
- Add comprehensive shape validation and error handling.
- Add gradient-checking and regression test coverage.
