import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product
from typing import Any


def binary_classification(model: Any, data: pd.DataFrame, label: int, model_type: str):
    features, labels = data["features"], data["targets"]
    y = np.argmax(labels, axis=1)

    if model_type == "neural_network":
        X = tf.convert_to_tensor(features)
        logits = model(X)
        probs = tf.nn.softmax(logits)
        y_hat_class = probs[:, label]
    elif model_type == "svm":
        probs = model.predict_proba(features.values)
        y_hat_class = probs[:, label]

    y_class = np.where(y == label, 1, 0)

    return y_class, y_hat_class
