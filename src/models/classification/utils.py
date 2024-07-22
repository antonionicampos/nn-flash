import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Any, Dict

FEATURES_NAMES = [
    "zN2",
    "zCO2",
    "zC1",
    "zC2",
    "zC3",
    "zIC4",
    "zNC4",
    "zIC5",
    "zNC5",
    "zC6",
    "zC7",
    "zC8",
    "zC9",
    "zC10",
    "zC11",
    "zC12",
    "zC13",
    "zC14",
    "zC15",
    "zC16",
    "zC17",
    "zC18",
    "zC19",
    "zC20",
    "P",
    "T",
]


def preprocessing(data):
    processed_data = data.copy()
    processed_data[FEATURES_NAMES[:-2]] = processed_data[FEATURES_NAMES[:-2]] / 100.0

    # P_sample (min, max): (10, 450)
    # T_sample (min, max): (150, 1125)
    processed_data["P"] = (processed_data["P"] - 10.0) / (450.0 - 10.0)
    processed_data["T"] = (processed_data["T"] - 150.0) / (1125.0 - 150.0)

    features = processed_data[FEATURES_NAMES].copy()
    labels = pd.get_dummies(processed_data["class"], dtype=np.float32)
    return features, labels


def save_results(results):
    pass


def training_history(results: List[List[Dict[str, Any]]], model_id: int):
    histories = [r["history"] for r in results[model_id]]
    f, axs = plt.subplots(len(histories), 2, figsize=(7, 15), sharex=True)

    for i, history in enumerate(histories):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        acc = history.history["categorical_accuracy"]
        val_acc = history.history["val_categorical_accuracy"]
        actual_epochs = np.arange(1, len(loss) + 1)

        axs[i, 0].plot(actual_epochs, loss, label="loss")
        axs[i, 0].plot(actual_epochs, val_loss, label="val_loss")
        axs[i, 0].legend(prop={"size": 8})
        axs[i, 0].grid()
        axs[i, 0].set_ylabel("Entropia Cruzada")

        axs[i, 1].plot(actual_epochs, acc, label="loss")
        axs[i, 1].plot(actual_epochs, val_acc, label="val_loss")
        axs[i, 1].legend(prop={"size": 8})
        axs[i, 1].grid()
        axs[i, 1].axhline(1.0, color="green")
        axs[i, 1].set_ylabel("Exatidão")

    f.tight_layout()
    plt.show()


def model_parameters_size(model: tf.keras.Model):
    parameters = [params.numpy().flatten().shape[0] for params in model.trainable_variables]
    return np.prod(parameters)
