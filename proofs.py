import os
import numpy as np
import matplotlib.pyplot as plt

from dense.dense_layer import Dense_Layer
from sequential.sequential import Sequential
from loss.binary_cross_entropy import Loss_BinaryLossEntropy
from loss.categorical_cross_entropy import Loss_CategoricalCrossEntropy
from loss.mse import Loss_MSE


def standardize_features(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True) + 1e-8
    return (X - mu) / sigma, mu, sigma


def normalize_target(y):
    mu = np.mean(y, axis=0, keepdims=True)
    sigma = np.std(y, axis=0, keepdims=True) + 1e-8
    return (y - mu) / sigma, mu, sigma


def make_binary_dataset(samples_per_class=700, noise=0.09, seed=42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, samples_per_class)

    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1 - np.cos(theta), -np.sin(theta) - 0.5])

    upper += rng.normal(0, noise, size=upper.shape)
    lower += rng.normal(0, noise, size=lower.shape)

    X = np.vstack([upper, lower])
    y = np.array([0] * samples_per_class + [1] * samples_per_class)
    return X, y


def make_categorical_dataset(samples_per_class=600, seed=42):
    rng = np.random.default_rng(seed)
    cov0 = np.array([[1.4, 0.7], [0.7, 1.1]])
    cov1 = np.array([[1.1, -0.6], [-0.6, 1.3]])
    cov2 = np.array([[1.3, 0.5], [0.5, 1.2]])
    c0 = rng.multivariate_normal(mean=[-2.8, 1.6], cov=cov0, size=samples_per_class)
    c1 = rng.multivariate_normal(mean=[2.7, 1.8], cov=cov1, size=samples_per_class)
    c2 = rng.multivariate_normal(mean=[0.2, -2.6], cov=cov2, size=samples_per_class)
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * samples_per_class + [1] * samples_per_class + [2] * samples_per_class)
    shuffle_idx = rng.permutation(X.shape[0])
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    return X, y


def make_regression_dataset(samples=900, seed=42):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.2, 3.2, samples).reshape(-1, 1)
    noise = rng.normal(loc=0.0, scale=0.07, size=(samples, 1))
    y = (0.65 * np.sin(2.1 * X)) + (0.22 * (X**3)) - (0.35 * X) + noise
    return X, y


def _plot_binary_boundary(ax, X_raw, y, probs, title):
    x_min, x_max = X_raw[:, 0].min() - 0.8, X_raw[:, 0].max() + 0.8
    y_min, y_max = X_raw[:, 1].min() - 0.8, X_raw[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    probs = probs.reshape(xx.shape)
    region = (probs >= 0.5).astype(int)

    ax.contourf(xx, yy, region, levels=1, alpha=0.25, cmap="coolwarm")
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.2)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y, cmap="coolwarm", s=9, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def _plot_categorical_boundary(ax, X_raw, y, class_map, title):
    x_min, x_max = X_raw[:, 0].min() - 0.8, X_raw[:, 0].max() + 0.8
    y_min, y_max = X_raw[:, 1].min() - 0.8, X_raw[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    class_map = class_map.reshape(xx.shape)

    ax.contourf(xx, yy, class_map, alpha=0.25, cmap="viridis")
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y, cmap="viridis", s=8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def train_binary():
    X_raw, y = make_binary_dataset()
    X, mu, sigma = standardize_features(X_raw)
    model = Sequential()
    model.add(Dense_Layer(2, 32, activation_fn="tanh"))
    model.add(Dense_Layer(32, 32, activation_fn="tanh"))
    model.add(Dense_Layer(32, 1, activation_fn="sigmoid"))
    model.compile(loss=Loss_BinaryLossEntropy(), lr=0.01)

    x_min, x_max = X_raw[:, 0].min() - 0.8, X_raw[:, 0].max() + 0.8
    y_min, y_max = X_raw[:, 1].min() - 0.8, X_raw[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid = (grid_raw - mu) / sigma

    y_prob_before = model.predict(X).reshape(-1)
    y_pred_before = (y_prob_before >= 0.5).astype(int)
    acc_before = np.mean(y_pred_before == y)
    boundary_before = model.predict(grid).reshape(-1)

    history = model.fit(X, y, epoch=14000, verbose=False)

    y_prob = model.predict(X).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    acc_after = np.mean(y_pred == y)
    boundary_after = model.predict(grid).reshape(-1)
    return X_raw, y, history, acc_before, acc_after, boundary_before, boundary_after


def train_categorical():
    X_raw, y = make_categorical_dataset()
    X, mu, sigma = standardize_features(X_raw)
    model = Sequential()
    model.add(Dense_Layer(2, 96, activation_fn="relu"))
    model.add(Dense_Layer(96, 64, activation_fn="relu"))
    model.add(Dense_Layer(64, 3, activation_fn="softmax"))
    model.compile(loss=Loss_CategoricalCrossEntropy(), lr=0.015)

    x_min, x_max = X_raw[:, 0].min() - 0.8, X_raw[:, 0].max() + 0.8
    y_min, y_max = X_raw[:, 1].min() - 0.8, X_raw[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid = (grid_raw - mu) / sigma

    y_prob_before = model.predict(X)
    y_pred_before = np.argmax(y_prob_before, axis=1)
    acc_before = np.mean(y_pred_before == y)
    boundary_before = np.argmax(model.predict(grid), axis=1)

    history = model.fit(X, y, epoch=9000, verbose=False)

    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)
    acc_after = np.mean(y_pred == y)
    boundary_after = np.argmax(model.predict(grid), axis=1)
    return X_raw, y, history, acc_before, acc_after, boundary_before, boundary_after


def train_regression():
    X_raw, y_raw = make_regression_dataset()
    X, x_mu, x_sigma = standardize_features(X_raw)
    y, y_mu, y_sigma = normalize_target(y_raw)
    model = Sequential()
    model.add(Dense_Layer(1, 96, activation_fn="tanh"))
    model.add(Dense_Layer(96, 64, activation_fn="relu"))
    model.add(Dense_Layer(64, 1, activation_required=False))
    model.compile(loss=Loss_MSE(), lr=0.008)

    y_pred_before = model.predict(X)
    mse_before = np.mean((y_pred_before - y) ** 2)

    history = model.fit(X, y, epoch=10000, verbose=False)

    y_pred = model.predict(X)
    mse_after = np.mean((y_pred - y) ** 2)

    y_before_raw = (y_pred_before * y_sigma) + y_mu
    y_after_raw = (y_pred * y_sigma) + y_mu
    mse_before_raw = np.mean((y_before_raw - y_raw) ** 2)
    mse_after_raw = np.mean((y_after_raw - y_raw) ** 2)
    return X_raw, y_raw, y_before_raw, y_after_raw, history, mse_before_raw, mse_after_raw


def build_proof_plots(output_path="proof_results.png"):
    X_bin, y_bin, h_bin, acc_bin_before, acc_bin_after, bin_before, bin_after = train_binary()
    X_cat, y_cat, h_cat, acc_cat_before, acc_cat_after, cat_before, cat_after = train_categorical()
    X_reg, y_reg, y_reg_before, y_reg_after, h_reg, mse_before, mse_after = train_regression()

    fig, axes = plt.subplots(3, 3, figsize=(19, 16))

    # Binary: before, after, loss
    _plot_binary_boundary(
        axes[0, 0], X_bin, y_bin, bin_before, f"Binary Before Training (acc={acc_bin_before:.3f})"
    )
    _plot_binary_boundary(
        axes[0, 1], X_bin, y_bin, bin_after, f"Binary After Training (acc={acc_bin_after:.3f})"
    )

    ax = axes[0, 2]
    ax.plot(h_bin, color="tab:blue", linewidth=1.8)
    ax.set_title("Binary Classification Loss (BCE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    # Categorical: before, after, loss
    _plot_categorical_boundary(
        axes[1, 0],
        X_cat,
        y_cat,
        cat_before,
        f"Multi-Class Before Training (acc={acc_cat_before:.3f})",
    )
    _plot_categorical_boundary(
        axes[1, 1], X_cat, y_cat, cat_after, f"Multi-Class After Training (acc={acc_cat_after:.3f})"
    )

    ax = axes[1, 2]
    ax.plot(h_cat, color="tab:green", linewidth=1.8)
    ax.set_title("Multi-Class Loss (Categorical CE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    # Regression: before, after, loss
    ax = axes[2, 0]
    ax.scatter(X_reg[:, 0], y_reg[:, 0], color="gray", s=6, alpha=0.45, label="Data")
    order = np.argsort(X_reg[:, 0])
    ax.plot(X_reg[order, 0], y_reg_before[order, 0], color="tab:blue", linewidth=2.0, label="Before")
    ax.set_title(f"Regression Before Training (MSE={mse_before:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    ax = axes[2, 1]
    ax.scatter(X_reg[:, 0], y_reg[:, 0], color="gray", s=8, alpha=0.45, label="Data")
    ax.plot(X_reg[order, 0], y_reg_after[order, 0], color="tab:red", linewidth=2.0, label="After")
    ax.set_title(f"Regression After Training (MSE={mse_after:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    ax = axes[2, 2]
    ax.plot(h_reg, color="tab:red", linewidth=1.8)
    ax.set_title("Regression Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    fig.suptitle("ndk-nn Proofs: Before vs After on Hard Datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=180)
    plt.show()


def main():
    output_file = "proof_results.png"
    build_proof_plots(output_path=output_file)
    print(f"Saved proof graph to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
