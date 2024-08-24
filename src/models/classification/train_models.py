import glob
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from src.data.handlers import DataLoader
from src.models.classification import NeuralNetClassifier
from src.models.classification.experiments import hparams
from src.utils import load_model_hparams
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)
np.random.seed(13)
tf.random.set_seed(13)


class ClassificationTraining:

    def __init__(self, samples_per_composition: int):
        self.samples_per_composition = samples_per_composition
        self.logger = logging.getLogger(__name__)
        self.results_folder = os.path.join(
            "data",
            "models",
            "classification",
            "saved_models",
            f"{samples_per_composition:03d}points",
        )

    def run(self):
        """Train classification models defined on models_specs.py script

        Results format:

        results = {
            "samples_per_composition": int,
            "outputs": [
                {
                    "model_id": int,
                    "model_name": str,
                    "arch": {"hidden_units": List[int], "activation": str},
                    "opt": {"lr": float, "epochs": int, "batch_size": int},
                    "folds": [
                        {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        data_loader = DataLoader()
        cv_data, _ = data_loader.load_cross_validation_datasets(
            problem="classification",
            samples_per_composition=self.samples_per_composition,
        )

        train_data, valid_data = cv_data["train"], cv_data["valid"]

        results = {"samples_per_composition": self.samples_per_composition, "outputs": []}
        training_start = datetime.now()
        for hp in load_model_hparams(hparams):
            model_name = hp["model_name"]
            model_type = hp["model_type"]
            params = hp["params"]

            model_results = {**hp}
            folds = []

            if model_type == "svm":
                print(f"\nModel: {model_name}")
                print(f"    Hyperparameters: {params}")

                self.logger.info(f"Model: {model_name}")
                self.logger.info(f"Hyperparameters: {params}")

                training_model_start = datetime.now()
                pbar = tqdm(total=len(train_data))
                for fold, (train, valid) in enumerate(zip(train_data, valid_data)):
                    pbar.set_description(f"Train using fold {fold+1} dataset")

                    train_features, train_labels = train["features"].values, train["targets"].values
                    valid_features, valid_labels = valid["features"].values, valid["targets"].values

                    train_labels = train_labels.argmax(axis=1)
                    valid_labels = valid_labels.argmax(axis=1)

                    model = SVC(probability=True, **params)
                    model.fit(train_features, train_labels)
                    probs = model.predict_proba(valid_features)
                    valid_labels_hat = np.argmax(probs, axis=1)
                    
                    folds.append({"fold": fold + 1, "model": model})
                    valid_accuracy = accuracy_score(valid_labels, valid_labels_hat)
                    self.logger.info(f"Fold {fold+1} dataset, valid accuracy: {valid_accuracy:.4f}")
                    pbar.update()

            elif model_type == "neural_network":
                opt_params = hp["opt"]

                learning_rate = opt_params["lr"]
                epochs = opt_params["epochs"]
                batch_size = opt_params["batch_size"]

                print(f"\nModel: {model_name}")
                print(f"    Architecture Params: {params}")
                print(f"    Optimization Params: {opt_params}", end="\n\n")

                self.logger.info(f"Model: {model_name}")
                self.logger.info(f"Archtecture Params: {params}")
                self.logger.info(f"Optimization Params: {opt_params}")

                training_model_start = datetime.now()
                pbar = tqdm(total=len(train_data))
                for fold, (train, valid) in enumerate(zip(train_data, valid_data)):
                    pbar.set_description(f"Training using fold {fold+1} dataset")

                    train_features, train_labels = train["features"], train["targets"]
                    valid_features, valid_labels = valid["features"], valid["targets"]

                    features, labels = train_features.values, train_labels.values
                    train_ds = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(10000).batch(batch_size)

                    features, labels = valid_features.values, valid_labels.values
                    valid_ds = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

                    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                    accuracy = tf.keras.metrics.CategoricalAccuracy()
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    callbacks = [
                        tf.keras.callbacks.ReduceLROnPlateau(),
                        # patience 10% of epochs size
                        tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=50),
                    ]
                    model = NeuralNetClassifier(**params)
                    model.compile(optimizer=optimizer, loss=loss_object, metrics=[accuracy])
                    h = model.fit(
                        train_ds,
                        epochs=epochs,
                        validation_data=valid_ds,
                        callbacks=callbacks,
                        verbose=0,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
                    )
                    folds.append({"fold": fold + 1, "model": model, "history": pd.DataFrame(h.history)})
                    val_results = {k: np.round(v[-1], decimals=4) for k, v in h.history.items() if "val_" in k}
                    self.logger.info(f"Fold {fold+1} dataset, {val_results}")
                    pbar.update()

            pbar.close()
            os.system("cls" if os.name == "nt" else "clear")
            model_results["folds"] = folds
            results["outputs"].append(model_results)
            self.logger.info(f"{model_name} training elapsed Time: {datetime.now() - training_model_start}")

        self.logger.info("Saving models")
        self.logger.info(f"Total elapsed Time: {datetime.now() - training_start}")
        self.save_training_models(results)

    def training_history(self, model_id: int):
        results = self.load_training_models()
        outputs = results["outputs"]
        model_results = list(filter(lambda item: item["model_id"] == model_id, outputs))[0]

        histories = [r["history"] for r in model_results["folds"]]
        f, axs = plt.subplots(len(histories), 2, figsize=(7, 20), sharex=True)

        for i, history in enumerate(histories):
            loss, val_loss = history["loss"], history["val_loss"]
            acc, val_acc = history["categorical_accuracy"], history["val_categorical_accuracy"]
            actual_epochs = np.arange(1, len(loss) + 1)

            axs[i, 0].plot(actual_epochs, loss, label="loss")
            axs[i, 0].plot(actual_epochs, val_loss, label="val_loss")
            axs[i, 0].legend(prop={"size": 8})
            axs[i, 0].grid()
            axs[i, 0].set_ylabel("Cross-Entropy")

            axs[i, 1].plot(actual_epochs, acc, label="loss")
            axs[i, 1].plot(actual_epochs, val_acc, label="val_loss")
            axs[i, 1].legend(prop={"size": 8})
            axs[i, 1].grid()
            axs[i, 1].axhline(1.0, color="green")
            axs[i, 1].set_ylabel("Accuracy")

        f.tight_layout()
        plt.show()

    def load_pickle(self, filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save_pickle(self, filepath, obj):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    def save_training_models(self, results):
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
                        "model_type": "neural_network" | "svm" | "random_forest",
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
        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        for output in results["outputs"]:
            model_folder = os.path.join(self.results_folder, output["model_name"])
            if not os.path.isdir(model_folder):
                os.mkdir(model_folder)

            folds = output.pop("folds")

            # Saving "id", "arch", "opt" objects
            self.save_pickle(os.path.join(model_folder, "model_info.pickle"), output)

            for fold_results in folds:
                fold = fold_results["fold"]
                model = fold_results["model"]

                fold_folder = os.path.join(model_folder, f"Fold{fold}")
                if not os.path.isdir(fold_folder):
                    os.mkdir(fold_folder)

                if output["model_type"] == "neural_network":
                    # Saving tf.keras.callbacks.History object to CSV file
                    history = fold_results["history"]
                    history.to_csv(os.path.join(fold_folder, "history.csv"), index=False)

                    # Saving tf.keras.Model weights
                    model.save(os.path.join(fold_folder, "model.keras"))
                elif output["model_type"] == "svm":
                    with open(os.path.join(fold_folder, "model.joblib"), "wb") as f:
                        joblib.dump(model, f)

    def load_training_models(self):
        """Load classification models training results"""
        n_folds = 10
        results = {"samples_per_composition": self.samples_per_composition}

        model_results = []
        for folder in glob.glob(os.path.join(self.results_folder, "*")):
            folds = []
            model_obj = self.load_pickle(os.path.join(folder, "model_info.pickle"))

            for fold in np.arange(n_folds):
                fold_folder = os.path.join(folder, f"Fold{fold+1}")
                if model_obj["model_type"] == "neural_network":
                    model = tf.keras.models.load_model(os.path.join(fold_folder, "model.keras"))
                    history = pd.read_csv(os.path.join(fold_folder, "history.csv"))
                    folds.append({"fold": fold + 1, "history": history, "model": model})
                elif model_obj["model_type"] == "svm":
                    with open(os.path.join(fold_folder, "model.joblib"), "rb") as f:
                        model = joblib.load(f)
                    folds.append({"fold": fold + 1, "history": history, "model": model})

            model_results.append({"folds": folds, **model_obj})
        results["outputs"] = sorted(model_results, key=lambda item: item["model_id"])
        return results
