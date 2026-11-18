"""
Multi-Layer Perceptron (MLP)
============================
A full neural network with multiple layers.

Architecture:
  Input Layer → [Hidden Layer 1] → [Hidden Layer 2] → ... → Output Layer

Each layer does:
  z = W·x + b        (linear combination)
  a = activation(z)  (non-linear transformation)

Training uses:
  1. Forward propagation  — compute predictions
  2. Loss computation     — measure error
  3. Backpropagation      — compute gradients
  4. Weight update        — gradient descent
"""

import numpy as np
from utils.activation import get_activation, get_derivative
from utils.loss import get_loss


class MLP:
    def __init__(
        self,
        layer_sizes,           # e.g. [2, 4, 4, 1] means 2 inputs, 2 hidden layers of 4, 1 output
        activations,           # e.g. ["relu", "relu", "sigmoid"]
        loss="binary_cross_entropy",
        learning_rate=0.01,
        epochs=100,
        l2_lambda=0.0,         # L2 regularization strength (0 = off)
        early_stopping=False,
        patience=10,
        dropout_rate=0.0,      # Dropout (0 = off)
    ):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization (works well with ReLU)."""
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            W = np.random.randn(fan_in, self.layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    # ─── Forward Propagation ─────────────────────────────────────────────────

    def _forward(self, X, training=False):
        """
        Pass data through all layers.
        Returns all intermediate values (needed for backprop).
        """
        self.z_cache = []   # raw linear outputs per layer
        self.a_cache = [X]  # activated outputs per layer (starts with input)

        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self.z_cache.append(z)

            act_fn = get_activation(self.activations[i])
            a = act_fn(z)

            # Apply dropout only during training (not evaluation)
            if training and self.dropout_rate > 0 and i < len(self.weights) - 1:
                mask = (np.random.rand(*a.shape) > self.dropout_rate)
                a = a * mask / (1 - self.dropout_rate)

            self.a_cache.append(a)

        return a  # final output = prediction

    # ─── Backpropagation ─────────────────────────────────────────────────────

    def _backward(self, X, y):
        """
        Compute gradients of loss w.r.t. every weight and bias.
        Uses chain rule layer by layer (from output back to input).
        """
        m = X.shape[0]  # number of samples
        loss_fn, loss_deriv = get_loss(self.loss_name)
        y_pred = self.a_cache[-1]

        # Gradient from loss
        delta = loss_deriv(y, y_pred)

        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        for i in reversed(range(len(self.weights))):
            act_name = self.activations[i]

            # For softmax + cross-entropy, delta is already the combined gradient
            if act_name != "softmax":
                deriv_fn = get_derivative(act_name)
                delta = delta * deriv_fn(self.z_cache[i])

            grads_W[i] = self.a_cache[i].T @ delta + self.l2_lambda * self.weights[i]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = delta @ self.weights[i].T

        return grads_W, grads_b

    # ─── Training ────────────────────────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None):
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        loss_fn, _ = get_loss(self.loss_name)
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            # Forward + backward
            y_pred = self._forward(X, training=True)
            grads_W, grads_b = self._backward(X, y)

            # Update weights (gradient descent)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_W[i]
                self.biases[i] -= self.lr * grads_b[i]

            # Record training metrics
            train_loss = float(loss_fn(y, y_pred))
            train_acc = self._compute_accuracy(y, y_pred)
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)

            # Validation metrics
            if X_val is not None:
                val_pred = self._forward(X_val, training=False)
                val_loss = float(loss_fn(y_val, val_pred))
                val_acc = self._compute_accuracy(y_val, val_pred)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

                # Early stopping
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        return self.history

    def predict(self, X):
        return self._forward(X, training=False)

    def _compute_accuracy(self, y_true, y_pred):
        if self.loss_name == "categorical_cross_entropy":
            return float(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)))
        elif self.loss_name == "binary_cross_entropy":
            return float(np.mean((y_pred >= 0.5).astype(int).flatten() == y_true.flatten().astype(int)))
        return 0.0  # regression doesn't use accuracy
