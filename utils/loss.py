"""
Loss Functions (Cost Functions)
================================
These measure how wrong the model's predictions are.
The goal of training is to minimize the loss.
"""

import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error — used for regression problems."""
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    """Gradient of MSE."""
    return 2 * (y_pred - y_true) / y_true.shape[0]


def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy — for binary classification (0 or 1)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true, y_pred):
    """Gradient of Binary Cross-Entropy."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross-Entropy — for multi-class classification."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_derivative(y_true, y_pred):
    """Gradient of Categorical Cross-Entropy."""
    return (y_pred - y_true) / y_true.shape[0]


LOSSES = {
    "mse": (mse, mse_derivative),
    "binary_cross_entropy": (binary_cross_entropy, binary_cross_entropy_derivative),
    "categorical_cross_entropy": (categorical_cross_entropy, categorical_cross_entropy_derivative),
}


def get_loss(name):
    return LOSSES[name]
