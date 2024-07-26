import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from typing import List, Any, Dict, Tuple
from scipy.stats import gmean
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, RocCurveDisplay, auc
from src.models.classification import NeuralNetClassifier
from src.utils.constants import FEATURES_NAMES, P_MIN_MAX, T_MIN_MAX, TARGET_NAMES


# Training functions
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
            "samples_per_composition": int,
            "outputs": [
                {
                    "model_id": int,
                    "model_name": str,
                    "arch": {
                        "hidden_units": List[int],
                        "activation": str,
                    },
                    "opt": {"lr": float, "epochs": int, "batch_size": int},
                    "folds": [
                        {
                            "fold": int,
                            "model": tf.keras.Model,
                            "history": tf.keras.callbacks.History
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    samples_per_composition = results.pop("samples_per_composition")
    results_folder = os.path.join(
        "src",
        "models",
        "classification",
        "saved_models",
        f"{samples_per_composition:03d}points",
    )
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    for output in results["outputs"]:
        model_folder = os.path.join(results_folder, output["model_name"])
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        folds = output.pop("folds")

        # Saving "id", "arch", "opt" objects
        save_pickle(os.path.join(model_folder, "model_info.pickle"), output)

        for fold_results in folds:
            fold = fold_results["fold"]
            history = fold_results["history"]
            model = fold_results["model"]

            fold_folder = os.path.join(model_folder, f"Fold{fold}")
            os.mkdir(fold_folder)

            # Saving tf.keras.callbacks.History object to CSV file
            history.to_csv(os.path.join(fold_folder, "history.csv"), index=False)

            # Saving tf.keras.Model weights
            model.save(os.path.join(fold_folder, "model.keras"))


def load_training_models(samples_per_composition: int):
    """Load classification models training results"""
    n_folds = 10
    results = {"samples_per_composition": samples_per_composition}
    results_folder = os.path.join(
        "src",
        "models",
        "classification",
        "saved_models",
        f"{samples_per_composition:03d}points",
    )

    model_results = []
    for folder in glob.glob(os.path.join(results_folder, "*")):
        folds = []
        model_obj = load_pickle(os.path.join(folder, "model_info.pickle"))

        for fold in np.arange(n_folds):
            model = tf.keras.models.load_model(os.path.join(folder, f"Fold{fold+1}", "model.keras"))
            history = pd.read_csv(os.path.join(folder, f"Fold{fold+1}", "history.csv"))
            folds.append({"fold": fold + 1, "history": history, "model": model})

        model_results.append({"folds": folds, **model_obj})

    results["outputs"] = sorted(model_results, key=lambda item: item["model_id"])
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


def binary_classification(model: tf.keras.Model, data: pd.DataFrame, label: int):
    features, labels = preprocessing(data)

    X = tf.convert_to_tensor(features)
    y = tf.convert_to_tensor(labels)
    y = tf.argmax(y, axis=1)

    logits = model(X)
    probs = tf.nn.softmax(logits)
    # y_hat = tf.argmax(probs, axis=1)

    y_class = np.where(y == label, 1, 0)
    y_hat_class = probs[:, label]

    return y_class, y_hat_class


def roc_analysis(
    results: List[List[Dict[str, Any]]],
    valid_files: List[str],
    model_id: int,
    figsize: Tuple[int] = (15, 5),
    xlim: Tuple[float] = (),
    ylim: Tuple[float] = (),
):
    n_folds = len(results[model_id])

    f, axs = plt.subplots(1, 3, figsize=figsize)

    for label in range(3):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold, data in enumerate(results[model_id]):
            model = data["model"]

            data_file = valid_files[fold]
            y, y_hat = binary_classification(model, data_file, label=label)

            viz = RocCurveDisplay.from_predictions(
                y,
                y_hat,
                name=f"ROC fold {fold + 1}",
                alpha=0.3,
                lw=1,
                ax=axs[label],
                plot_chance_level=(fold == n_folds - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs[label].step(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axs[label].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm 1 \sigma$",
        )

        axs[label].set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label '{TARGET_NAMES[label]}')",
        )
        axs[label].legend(loc="lower right", prop={"size": 8})
        axs[label].grid()

        if xlim:
            axs[label].set_xlim(xlim)

        if ylim:
            axs[label].set_ylim(ylim)
    f.tight_layout()
    plt.show()
