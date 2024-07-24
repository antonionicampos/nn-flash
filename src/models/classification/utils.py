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

    for model_name, model_results in results.items():
        model_folder = os.path.join(results_folder, model_name)
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        folds = model_results.pop("folds")

        # Saving "id", "arch", "opt" objects
        save_pickle(os.path.join(model_folder, f"model_info.pickle"), model_results)

        for fold_results in folds:
            fold = fold_results["fold"]
            history = fold_results["history"]
            model = fold_results["model"]

            # Saving tf.keras.callbacks.History object to CSV file
            history.to_csv(os.path.join(model_folder, f"history_fold={fold}.csv"), index=False)

            # Saving tf.keras.Model object
            model.save_weights(os.path.join(model_folder, f"model_fold={fold}.weights.h5"))


def load_training_models(samples_per_composition: int):
    """Load classification models training results"""
    n_folds = 10
    results = {}
    results_folder = os.path.join(
        "src", "models", "classification", "saved_models", f"{samples_per_composition:03d}points"
    )

    for folder in glob.glob(os.path.join(results_folder, "*")):
        folds = []
        model_name = folder.split("\\")[-1]
        model_obj = load_pickle(os.path.join(folder, "model_info.pickle"))
        arch_params = model_obj["arch"]

        for fold in np.arange(n_folds):
            model = NeuralNetClassifier(**arch_params)
            model.load_weights(os.path.join(folder, f"model_fold={fold+1}.weights.h5"))
            history = pd.read_csv(os.path.join(folder, f"history_fold={fold+1}.csv"))
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


# Evaluation functions
def model_parameters_size(model: tf.keras.Model):
    parameters = [params.numpy().flatten().shape[0] for params in model.trainable_variables]
    return np.prod(parameters)


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


def confusion_matrix_plot(results: List[List[Dict[str, Any]]], valid_files: List[str]):
    pass


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

    cross_entropy_matrix = np.zeros((num_folds, num_models))
    bic = np.zeros((num_folds, num_models))
    aic = np.zeros((num_folds, num_models))
    for i, models in enumerate(results):
        for j, (result, valid_f) in enumerate(zip(models, valid_files)):
            valid_data = pd.read_csv(valid_f)
            valid_features, valid_labels = preprocessing(valid_data)

            X_valid = tf.convert_to_tensor(valid_features)
            y_valid = tf.convert_to_tensor(valid_labels)

            model = result["model"]
            logits = model(X_valid)
            probs = tf.nn.softmax(logits)

            # Cross-Entropy
            cross_entropy = tf.keras.losses.categorical_crossentropy(y_valid, probs)
            cross_entropy_matrix[j, i] = tf.reduce_mean(cross_entropy)

            # BIC (Bayesian Information Criterion)
            # AIC (Akaike Information Criterion)
            n_params = model_parameters_size(model)
            n_samples = valid_data.shape[0]
            likelihood = -tf.reduce_mean(cross_entropy)

            bic = n_params * np.log(n_samples) - 2 * np.log(likelihood)
            aic = 2 * n_params - 2 * np.log(likelihood)

    return {
        "accucary": accuracies,
        "sp_index": sp_indexes,
        "cross_entropy": cross_entropy_matrix,
        "bic": bic,
        "aic": aic,
    }


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
