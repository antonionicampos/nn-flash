import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from typing import List, Any, Dict
from src.models.classification import NeuralNetClassifier
from src.utils.constants import FEATURES_NAMES, P_MIN_MAX, T_MIN_MAX


def preprocessing(data):
    processed_data = data.copy()
    processed_data[FEATURES_NAMES[:-2]] = processed_data[FEATURES_NAMES[:-2]] / 100.0

    P_min, P_max = P_MIN_MAX
    T_min, T_max = T_MIN_MAX
    processed_data["P"] = (processed_data["P"] - P_min) / (P_max - P_min)
    processed_data["T"] = (processed_data["T"] - T_min) / (T_max - T_min)

    features = processed_data[FEATURES_NAMES].copy()
    labels = pd.get_dummies(processed_data["class"], dtype=np.float32)
    return features, labels


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def save_training_models(results):
    """Save classification models training results

    Parameters
    ----------
    results : dict
        Model training results structure. Format below:
        {
            "<MODEL_NAME>": {
                "id": <UNIQUE_ID>,
                "arch": {"activation": tf.keras.activations.Activation, "hidden_units": List[int]},
                "opt": {"batch_size": int, "epochs": int, "lr": float}
                "folds": [
                    {"fold": int, "history": tf.keras.callbacks.History, "model": tf.keras.Model}
                ]
            }
        }
    """
    results_folder = os.path.join("src", "models", "classification", "saved_models")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    for model_name, model_results in results.items():
        model_folder = os.path.join(results_folder, model_name)
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        folds = model_results.pop("folds")

        # Saving "id", "arch", "opt" objects
        with open(os.path.join(model_folder, f"model_obj.pickle"), "wb") as f:
            pickle.dump(model_results, f)

        for fold_results in folds:
            fold = fold_results["fold"]
            history = fold_results["history"]
            model = fold_results["model"]

            # Saving tf.keras.callbacks.History object
            with open(os.path.join(model_folder, f"history_fold={fold}.pickle"), "wb") as f:
                pickle.dump(history, f)

            # Saving tf.keras.Model object
            model.save_weights(os.path.join(model_folder, f"model_fold={fold}.weights.h5"))


def load_training_models():
    """Load classification models training results"""
    n_folds = 10
    results = {}
    results_folder = os.path.join("src", "models", "classification", "saved_models")

    for folder in glob.glob(os.path.join(results_folder, "*")):
        folds = []
        model_name = folder.split("\\")[-1]
        model_obj = load_pickle(os.path.join(folder, "model_obj.pickle"))
        arch_params = model_obj["arch"]

        for fold in np.arange(n_folds):
            model = NeuralNetClassifier(**arch_params)
            model.load_weights(os.path.join(folder, f"model_fold={fold+1}.weights.h5"))
            history = load_pickle(os.path.join(folder, f"history_fold={fold+1}.pickle"))
            folds.append({"fold": fold + 1, "history": history, "model": model})

        results[model_name] = {"folds": folds, **model_obj}

    return results


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
