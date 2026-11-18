"""
Flask Backend
=============
This is the web server that connects the UI to the neural network models.

Routes:
  GET  /              → serve the HTML page
  POST /train         → train a model, return results
  POST /predict       → predict on uploaded CSV
"""

import json
import numpy as np
import io
import csv

from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.perceptron import Perceptron
from models.mlp import MLP
from utils.metrics import (
    accuracy, confusion_matrix, precision_recall_f1,
    mse_metric, rmse_metric, r2_score
)

app = Flask(__name__)

# Store the last trained model in memory (simple approach)
_model_store = {}


# ─── Helper: Parse CSV upload ─────────────────────────────────────────────────

def parse_csv(file_content):
    """Read CSV bytes and return headers + rows."""
    text = file_content.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV")
    headers = rows[0]
    data = rows[1:]
    return headers, data


def prepare_data(headers, data, target_col, problem_type):
    """Convert CSV data into numpy arrays X, y."""
    # Build feature matrix
    all_rows = []
    for row in data:
        if len(row) == len(headers):
            all_rows.append(row)

    if not all_rows:
        raise ValueError("No valid rows found")

    target_idx = headers.index(target_col)
    feature_indices = [i for i in range(len(headers)) if i != target_idx]

    X_raw = [[row[i] for i in feature_indices] for row in all_rows]
    y_raw = [row[target_idx] for row in all_rows]

    # Convert X to float (handle non-numeric by label encoding)
    X = []
    for row in X_raw:
        encoded = []
        for val in row:
            try:
                encoded.append(float(val))
            except:
                encoded.append(hash(val) % 1000)
        X.append(encoded)
    X = np.array(X, dtype=float)

    # Convert y based on problem type
    if problem_type == "regression":
        y = np.array([float(v) for v in y_raw]).reshape(-1, 1)
    elif problem_type == "binary":
        # Try numeric first
        try:
            y_vals = np.array([float(v) for v in y_raw])
            unique = np.unique(y_vals)
            if len(unique) == 2:
                y = (y_vals == unique[1]).astype(float).reshape(-1, 1)
            else:
                raise ValueError
        except:
            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(float).reshape(-1, 1)
    else:  # multiclass
        try:
            y_vals = np.array([int(float(v)) for v in y_raw])
        except:
            le = LabelEncoder()
            y_vals = le.fit_transform(y_raw)
        num_classes = len(np.unique(y_vals))
        y = np.zeros((len(y_vals), num_classes))
        y[np.arange(len(y_vals)), y_vals] = 1

    return X, y


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    try:
        # ── 1. Parse form data ──
        file = request.files.get("dataset")
        config = json.loads(request.form.get("config", "{}"))

        problem_type = config.get("problem_type", "binary")
        model_type   = config.get("model_type", "mlp")
        target_col   = config.get("target_col", "")
        test_size    = float(config.get("test_size", 0.2))
        lr           = float(config.get("learning_rate", 0.01))
        epochs       = int(config.get("epochs", 100))
        hidden_layers = config.get("hidden_layers", [4])
        activation   = config.get("activation", "relu")
        out_activation = config.get("out_activation", "sigmoid")
        loss_fn      = config.get("loss", "binary_cross_entropy")
        l2_lambda    = float(config.get("l2_lambda", 0.0))
        early_stopping = bool(config.get("early_stopping", False))
        dropout_rate = float(config.get("dropout_rate", 0.0))

        # ── 2. Load & prepare data ──
        if file:
            headers, data = parse_csv(file.read())
            if not target_col or target_col not in headers:
                target_col = headers[-1]
            X, y = prepare_data(headers, data, target_col, problem_type)
        else:
            # Demo: XOR dataset if no file uploaded
            X = np.array([[0,0],[0,1],[1,0],[1,1]] * 25, dtype=float)
            y = np.array([[0],[1],[1],[0]] * 25, dtype=float)

        # ── 3. Scale features ──
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # ── 4. Split train/test ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # ── 5. Build model ──
        n_features = X.shape[1]
        if problem_type == "multiclass":
            n_output = y.shape[1]
        elif problem_type == "binary":
            n_output = 1
        else:
            n_output = 1

        if model_type == "perceptron":
            model = Perceptron(learning_rate=lr, epochs=epochs, activation="sigmoid")
            history = model.fit(X_train, y_train, X_test, y_test)

        else:  # MLP
            layer_sizes = [n_features] + hidden_layers + [n_output]
            activations = [activation] * len(hidden_layers) + [out_activation]

            model = MLP(
                layer_sizes=layer_sizes,
                activations=activations,
                loss=loss_fn,
                learning_rate=lr,
                epochs=epochs,
                l2_lambda=l2_lambda,
                early_stopping=early_stopping,
                patience=15,
                dropout_rate=dropout_rate,
            )
            history = model.fit(X_train, y_train, X_test, y_test)

        # ── 6. Evaluate ──
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        metrics = {}
        if problem_type == "regression":
            metrics = {
                "mse": mse_metric(y_test, y_pred_test),
                "rmse": rmse_metric(y_test, y_pred_test),
                "r2": r2_score(y_test, y_pred_test),
            }
        else:
            n_cls = y.shape[1] if y.ndim > 1 and y.shape[1] > 1 else 2
            prf = precision_recall_f1(y_test, y_pred_test)
            metrics = {
                "accuracy": accuracy(y_test, y_pred_test),
                "train_accuracy": accuracy(y_train, y_pred_train),
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "confusion_matrix": confusion_matrix(y_test, y_pred_test, num_classes=n_cls),
            }

        # ── 7. Decision boundary (only for 2D input) ──
        decision_boundary = None
        if X.shape[1] == 2 and problem_type != "regression":
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60),
                                 np.linspace(y_min, y_max, 60))
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = model.predict(grid)
            if zz.ndim > 1 and zz.shape[1] > 1:
                zz = np.argmax(zz, axis=1)
            else:
                zz = (zz >= 0.5).astype(int).flatten()

            decision_boundary = {
                "xx": xx.tolist(),
                "yy": yy.tolist(),
                "zz": zz.reshape(xx.shape).tolist(),
                "X_train": X_train.tolist(),
                "y_train": (np.argmax(y_train, axis=1) if y_train.ndim > 1 and y_train.shape[1] > 1
                            else y_train.flatten().astype(int).tolist()),
                "X_test": X_test.tolist(),
                "y_test": (np.argmax(y_test, axis=1) if y_test.ndim > 1 and y_test.shape[1] > 1
                           else y_test.flatten().astype(int).tolist()),
            }

        # Store model for reuse
        _model_store["model"] = model
        _model_store["scaler"] = scaler

        return jsonify({
            "success": True,
            "history": history,
            "metrics": metrics,
            "decision_boundary": decision_boundary,
            "n_samples": len(X),
            "n_features": n_features,
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/columns", methods=["POST"])
def get_columns():
    """Return column names from uploaded CSV."""
    try:
        file = request.files.get("dataset")
        if not file:
            return jsonify({"columns": []})
        headers, _ = parse_csv(file.read())
        return jsonify({"columns": headers})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
