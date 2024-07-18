import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from classification.training import preprocessing
from scipy.stats import gmean
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
    RocCurveDisplay,
    auc,
)
from typing import List, Dict, Any, Tuple


TARGET_NAMES: List[str] = ["Gas", "Mix", "Oil"]


def binary_classification(model: tf.keras.Model, data_file: str, label: int):
    data = pd.read_csv(data_file)
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


def confusion_matrix(results: List[List[Dict[str, Any]]], valid_files: List[str]):
    num_classes = 3
    num_models = len(results)
    num_folds = len(results[0])

    confusion_matrices = np.zeros((num_folds, num_models, num_classes, num_classes))
    for i, models in enumerate(results):
        for j, (result, valid_f) in enumerate(zip(models, valid_files)):
            valid_data = pd.read_csv(valid_f)
            valid_features, valid_labels = preprocessing(valid_data)

            X_valid = tf.convert_to_tensor(valid_features)
            y_valid = tf.convert_to_tensor(valid_labels)
            y_valid = tf.argmax(y_valid, axis=1)

            model = result["model"]
            logits = model(X_valid)
            probs = tf.nn.softmax(logits)

            y_valid_hat = tf.argmax(probs, axis=1)
            confusion_matrices[j, i] = sklearn_confusion_matrix(y_valid, y_valid_hat)

    return confusion_matrices


def performance_indices(results: List[List[Dict[str, Any]]], valid_files: List[str]):
    num_folds = len(valid_files)
    num_models = len(results)

    confusion_matrices = confusion_matrix(results, valid_files)
    accuracies = np.zeros((num_folds, num_models))
    sp_indexes = np.zeros((num_folds, num_models))

    for model in np.arange(confusion_matrices.shape[1]):
        for fold in np.arange(confusion_matrices.shape[0]):
            cm = confusion_matrices[fold, model, :, :]

            # Accuracy
            accuracies[fold, model] = np.diag(cm).sum() / cm.sum()

            # Sensitivity/Specificity

            # SP Index
            sensitivity = np.zeros([cm.shape[0]])
            for i in np.arange(cm.shape[0]):
                sensitivity[i] = cm[i, i] / cm[i, :].sum()
            sp_indexes[fold, model] = np.sqrt(np.mean(sensitivity) * gmean(sensitivity))

    return {"accucary": accuracies, "sp_index": sp_indexes}


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
