"""
Activation Functions
====================
These are the "switches" of a neural network.
Each function decides how strongly a neuron fires.
"""

import numpy as np


def sigmoid(z):
    """Squashes any value to a range between 0 and 1. Good for binary output."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """How much sigmoid changes at a given point (used in backprop)."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """ReLU: outputs 0 if negative, otherwise outputs z. Simple and powerful."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)


def tanh(z):
    """Tanh: squashes values between -1 and 1. Centered around 0."""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2


def softmax(z):
    """Softmax: converts raw scores into probabilities that sum to 1. Used for multi-class."""
    e = np.exp(z - np.max(z, axis=1, keepdims=True))  # subtract max for numerical stability
    return e / np.sum(e, axis=1, keepdims=True)


# Lookup tables so we can call functions by name (string)
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "softmax": softmax,
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "relu": relu_derivative,
    "tanh": tanh_derivative,
    "softmax": lambda z: np.ones_like(z),  # handled separately in backprop
}


def get_activation(name):
    return ACTIVATIONS[name]


def get_derivative(name):
    return DERIVATIVES[name]
