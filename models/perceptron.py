"""
Perceptron Model
================
The original neural network — invented in 1958 by Frank Rosenblatt.
It's just a single neuron: takes inputs, multiplies by weights, adds bias,
applies an activation, and outputs a prediction.

  Input → [weights · x + bias] → activation → output
"""

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100, activation="step"):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.weights = None
        self.bias = 0.0
        self.history = {"loss": [], "accuracy": []}

    def _activate(self, z):
        if self.activation == "step":
            return (z >= 0).astype(float)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return z

    def fit(self, X, y, X_val=None, y_val=None):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = {"loss": [], "accuracy": []}

        for epoch in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self._activate(z)

            # Update weights using perceptron rule
            error = y.flatten() - y_pred
            self.weights += self.lr * X.T @ error
            self.bias += self.lr * np.sum(error)

            # Track metrics
            loss = float(np.mean(error ** 2))
            acc = float(np.mean((y_pred >= 0.5).astype(int) == y.flatten().astype(int)))
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)

        return self.history

    def predict(self, X):
        z = X @ self.weights + self.bias
        return self._activate(z).reshape(-1, 1)
