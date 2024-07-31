import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from datetime import datetime
from src.data.handlers import DataLoader
from src.models.regression import NeuralNet, ResidualNeuralNet
from src.models.regression.models_specs import hparams
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)
np.random.seed(13)
tf.random.set_seed(13)


class RegressionTraining:

    def __init__(self, samples_per_composition: int):
        self.samples_per_composition = samples_per_composition
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Train regression models defined on models_specs.py script

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
        cv_data, min_max = data_loader.load_cross_validation_datasets(
            problem="regression",
            samples_per_composition=self.samples_per_composition,
        )

        train_data, valid_data = cv_data["train"], cv_data["valid"]

        results = {"samples_per_composition": self.samples_per_composition, "outputs": []}
        training_start = datetime.now()
        for hp in hparams:
            model_name = hp["model_name"]
            arch_params = hp["arch"]
            opt_params = hp["opt"]

            learning_rate = opt_params["lr"]
            epochs = opt_params["epochs"]
            batch_size = opt_params["batch_size"]

            model_type = arch_params.pop("type", "")
            model_results = {**hp}
            folds = []

            print(f"\nModel: {model_name}")
            print(f"    Archtecture Params: {arch_params}")
            print(f"    Optimization Params: {opt_params}", end="\n\n")

            self.logger.info(f"Model: {model_name}")
            self.logger.info(f"Archtecture Params: {arch_params}")
            self.logger.info(f"Optimization Params: {opt_params}")

            training_model_start = datetime.now()
            pbar = tqdm(total=len(train_data))
            for fold, (train, valid) in enumerate(zip(train_data, valid_data)):
                pbar.set_description(f"Train using fold {fold+1} dataset")

                train_features, train_labels = train["features"], train["targets"]
                valid_features, valid_labels = valid["features"], valid["targets"]

                features, labels = train_features.values, train_labels.values
                train_ds = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(10000).batch(batch_size)

                features, labels = valid_features.values, valid_labels.values
                valid_ds = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

                loss_object = tf.keras.losses.MeanSquaredError()
                mae = tf.keras.metrics.MeanAbsoluteError()
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                callbacks = [
                    tf.keras.callbacks.ReduceLROnPlateau(),
                    tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=10),
                ]

                if model_type == "residual":
                    model = ResidualNeuralNet(**arch_params)
                else:
                    model = NeuralNet(**arch_params)

                model.compile(optimizer=optimizer, loss=loss_object, metrics=[mae])
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
            os.system("cls")
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
            acc, val_acc = history["mean_absolute_error"], history["val_mean_absolute_error"]
            actual_epochs = np.arange(1, len(loss) + 1)

            axs[i, 0].plot(actual_epochs, loss, label="loss")
            axs[i, 0].plot(actual_epochs, val_loss, label="val_loss")
            axs[i, 0].legend(prop={"size": 8})
            axs[i, 0].grid()
            axs[i, 0].set_ylabel("Mean Squared Error")

            axs[i, 1].plot(actual_epochs, acc, label="loss")
            axs[i, 1].plot(actual_epochs, val_acc, label="val_loss")
            axs[i, 1].legend(prop={"size": 8})
            axs[i, 1].grid()
            axs[i, 1].set_ylabel("Mean Absolute Error")

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
        """Save regression models training results

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
            "regression",
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
            self.save_pickle(os.path.join(model_folder, "model_info.pickle"), output)

            for fold_results in folds:
                fold = fold_results["fold"]
                history = fold_results["history"]
                model = fold_results["model"]

                fold_folder = os.path.join(model_folder, f"Fold{fold}")
                if not os.path.isdir(fold_folder):
                    os.mkdir(fold_folder)

                # Saving tf.keras.callbacks.History object to CSV file
                history.to_csv(os.path.join(fold_folder, "history.csv"), index=False)

                # Saving tf.keras.Model weights
                model.save(os.path.join(fold_folder, "model.keras"))

    def load_training_models(self):
        """Load regression models training results"""
        n_folds = 10
        results = {"samples_per_composition": self.samples_per_composition}
        results_folder = os.path.join(
            "src",
            "models",
            "regression",
            "saved_models",
            f"{self.samples_per_composition:03d}points",
        )

        model_results = []
        for folder in glob.glob(os.path.join(results_folder, "*")):
            folds = []
            model_obj = self.load_pickle(os.path.join(folder, "model_info.pickle"))

            for fold in np.arange(n_folds):
                model = tf.keras.models.load_model(os.path.join(folder, f"Fold{fold+1}", "model.keras"))
                history = pd.read_csv(os.path.join(folder, f"Fold{fold+1}", "history.csv"))
                folds.append({"fold": fold + 1, "history": history, "model": model})

            model_results.append({"folds": folds, **model_obj})

        results["outputs"] = sorted(model_results, key=lambda item: item["model_id"])
        return results
