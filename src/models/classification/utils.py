import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import preprocessing


def model_parameters_size(model: tf.keras.Model):
    parameters = [params.numpy().flatten().shape[0] for params in model.trainable_variables]
    return np.prod(parameters)


def binary_classification(model: tf.keras.Model, data: pd.DataFrame, label: int):
    features, labels = preprocessing(data, problem="classification")
    X = tf.convert_to_tensor(features)
    y = tf.convert_to_tensor(labels)
    y = tf.argmax(y, axis=1)

    logits = model(X)
    probs = tf.nn.softmax(logits)
    y_class = np.where(y == label, 1, 0)
    y_hat_class = probs[:, label]
    return y_class, y_hat_class
