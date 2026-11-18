"""
Evaluation Metrics
==================
Used after training to measure how well the model performs.
"""

import numpy as np


# ─── Classification Metrics ───────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    return np.mean((y_pred >= 0.5).astype(int).flatten() == y_true.flatten())


def confusion_matrix(y_true, y_pred, num_classes=2):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_pred_labels = (y_pred >= 0.5).astype(int).flatten()
        y_true_labels = y_true.flatten().astype(int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_labels, y_pred_labels):
        if t < num_classes and p < num_classes:
            cm[t][p] += 1
    return cm.tolist()


def precision_recall_f1(y_true, y_pred):
    """Returns macro-averaged precision, recall, F1."""
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        num_classes = y_pred.shape[1]
    else:
        y_pred_labels = (y_pred >= 0.5).astype(int).flatten()
        y_true_labels = y_true.flatten().astype(int)
        num_classes = 2

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((y_pred_labels == c) & (y_true_labels == c))
        fp = np.sum((y_pred_labels == c) & (y_true_labels != c))
        fn = np.sum((y_pred_labels != c) & (y_true_labels == c))
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f = 2 * p * r / (p + r + 1e-10)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ─── Regression Metrics ───────────────────────────────────────────────────────

def mse_metric(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))
