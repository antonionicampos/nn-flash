import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from datetime import datetime
from itertools import product
from src.data.handlers import DataLoader
from src.models.regression import NeuralNet, ResidualNeuralNet, MeanSquaredErrorWithSoftConstraint
from src.models.regression.experiments import hparams
from src.utils import denorm, load_model_hparams
from src.utils.constants import K_FOLDS
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)
np.random.seed(13)
tf.random.set_seed(13)


class RegressionTraining:

    def __init__(self):
        self.k_fold = K_FOLDS
        self.logger = logging.getLogger(__name__)
        self.results_folder = os.path.join("data", "models", "regression", "saved_models")

    def run(self):
        """Train regression models defined on experiments.py script

        Results format:

        results = {
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
        cv_data, _ = data_loader.load_cross_validation_datasets(problem="regression")

        train_data, valid_data = cv_data["train"], cv_data["valid"]

        results = {"outputs": []}
        training_start = datetime.now()
        for hp in load_model_hparams(hparams):
            model_name = hp["model_name"]
            arch_params = hp["params"]
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
                    # patience 10% of epochs size
                    tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=50),
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
                val_results = {k: np.round(v[-1], decimals=5) for k, v in h.history.items() if "val_" in k}
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
                "outputs": [
                    {
                        "model_id": int,
                        "model_name": str,
                        "params": {
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
        results = {}

        model_results = []
        for folder in glob.glob(os.path.join(self.results_folder, "*")):
            folds = []
            model_obj = self.load_pickle(os.path.join(folder, "model_info.pickle"))

            for fold in np.arange(self.k_fold):
                model = tf.keras.models.load_model(os.path.join(folder, f"Fold{fold+1}", "model.keras"))
                history = pd.read_csv(os.path.join(folder, f"Fold{fold+1}", "history.csv"))
                folds.append({"fold": fold + 1, "history": history, "model": model})

            model_results.append({"folds": folds, **model_obj})

        results["outputs"] = sorted(model_results, key=lambda item: item["model_id"])
        return results

    def convert_K_to_XY(self, outputs, inputs):
        K = outputs[:, :-1]
        V = outputs[:, -1:]
        Z = inputs[:, :-2]
        L = 1 - V
        X = Z / (L + V * K)
        Y = K * X
        return X, Y

    def train_mse_loss_with_soft_constraint(self, log=False):
        """Training models using a modified MSE Loss with output composition constraint

        Results format:

        results = {
            "outputs": [
                {
                    "hparams": {"hidden_units": List[int], "activation": str},
                    "folds": [
                        {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
                        ...
                    ]
                },
                ...
            ]
        }"""
        params = {"hidden_layers": [3, 4, 5], "hidden_units": [128, 256, 512], "lambda": [0.0, 1e-5, 1e-3]}

        models_folder = os.path.join("data", "models", "regression_with_constrained_loss", "saved_results")
        results_folder = os.path.join("data", "models", "regression_with_constrained_loss", "saved_performance_indices")

        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)

        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        dl = DataLoader()
        datasets, minmax = dl.load_cross_validation_datasets(problem="regression")

        results = {"outputs": []}
        self.hyperparameters = list(product(*[vals for vals in params.values()]))
        for i, (hidden_units, neurons, lambda_) in enumerate(self.hyperparameters):
            hparams = {"hidden_units": [neurons for _ in range(hidden_units)], "lambda": lambda_}
            r = {"hparams": {**hparams}, "folds": []}
            start = datetime.now()

            hidden_units = hparams["hidden_units"]
            activation = "relu"
            batch_size = 32
            epochs = 500
            lr = 0.001
            lambda_ = hparams["lambda"]

            train_log = "Epoch: {:04d}, train loss: {:.5f}, valid loss: {:.5f}"

            for j, (train, valid, minmax_vals) in enumerate(zip(datasets["train"], datasets["valid"], minmax)):
                min_vals, max_vals = minmax_vals
                min_vals = tf.convert_to_tensor(min_vals, dtype=tf.float32)
                max_vals = tf.convert_to_tensor(max_vals, dtype=tf.float32)
                x_train = tf.convert_to_tensor(train["features"], dtype=tf.float32)
                y_train = tf.convert_to_tensor(train["targets"], dtype=tf.float32)
                x_valid = tf.convert_to_tensor(valid["features"], dtype=tf.float32)
                y_valid = tf.convert_to_tensor(valid["targets"], dtype=tf.float32)

                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

                model = NeuralNet(hidden_units, activation)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                loss_func = MeanSquaredErrorWithSoftConstraint(lambda_=lambda_)

                train_losses, valid_losses = [], []
                best_valid_loss = 1e3
                for epoch in range(epochs):
                    for x_batch_train, y_batch_train in train_dataset:

                        # Record operations
                        with tf.GradientTape() as tape:
                            y_hat = model(x_batch_train, training=True)
                            loss_val = loss_func(y_batch_train, y_hat, x_batch_train, min_vals, max_vals)

                        # Grads dloss/dwij
                        grads = tape.gradient(loss_val, model.trainable_weights)

                        # Optimizer using grads
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))

                        # Validation loss
                        loss_val = loss_func(y_batch_train, y_hat, x_batch_train, min_vals, max_vals)

                    y_hat_train = model(x_train)
                    y_hat_valid = model(x_valid)
                    train_loss = loss_func(y_train, y_hat_train, x_train, min_vals, max_vals)
                    valid_loss = loss_func(y_valid, y_hat_valid, x_valid, min_vals, max_vals)

                    train_losses.append(float(train_loss))
                    valid_losses.append(float(valid_loss))

                    if float(valid_loss) < best_valid_loss:
                        best_model_path = os.path.join(
                            models_folder,
                            f"best_model_model_id={i}_fold={j}_epoch={epoch}.keras",
                        )
                        best_epoch = epoch
                        best_model = tf.keras.models.clone_model(model)
                        best_model.set_weights(model.get_weights())
                        best_valid_loss = float(valid_loss)

                    if log and (epoch + 1) % 100 == 0:
                        self.logger.info(train_log.format(epoch + 1, float(train_loss), float(valid_loss)))

                tf.keras.models.save_model(best_model, best_model_path)
                y_hat_valid = best_model(x_valid)
                y_pred = denorm(y_hat_valid, min_vals, max_vals)
                xi_pred, yi_pred = self.convert_K_to_XY(y_pred, x_valid)

                self.logger.info(
                    f"{hparams}, fold: {j+1}, best valid loss: {round(best_valid_loss, 6)} [epoch {best_epoch}]"
                )

                r["folds"].append(
                    {
                        "fold": j + 1,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses,
                        "best_valid_loss": best_valid_loss,
                        "summ_xi_hat": xi_pred.numpy().sum(axis=-1).mean(),
                        "summ_yi_hat": yi_pred.numpy().sum(axis=-1).mean(),
                    }
                )

            end = datetime.now()
            results["outputs"].append(r)
            self.logger.info(f"training model: {i+1}/{len(self.hyperparameters)}, elapsed time: {end - start}")

        self.save_pickle(os.path.join(results_folder, "train_results.pickle"), results)

    def plot_mse_loss_with_soft_constraint(self):
        root_folder = os.path.join("data", "models", "regression_with_constrained_loss")
        viz_folder = os.path.join(root_folder, "saved_viz")
        results_filename = os.path.join(root_folder, "saved_performance_indices", "train_results.pickle")

        if not os.path.isdir(viz_folder):
            os.makedirs(viz_folder)

        results = self.load_pickle(results_filename)
        hparams = pd.DataFrame.from_records(
            [
                {
                    "hidden_layers": len(model["hparams"]["hidden_units"]),
                    "hidden_units": model["hparams"]["hidden_units"][0],
                    "lambda": model["hparams"]["lambda"],
                }
                for model in results["outputs"]
            ]
        ).astype({"hidden_units": "int16", "hidden_layers": "int16", "lambda": "float64"})
        hparams = hparams[hparams["lambda"] < 0.1].sort_values([2, 0, 1], axis=1)
        sorted_idx = hparams.index
        xy = np.array([[[f["summ_xi_hat"], f["summ_yi_hat"]] for f in m["folds"]] for m in results["outputs"]])
        losses = np.array([[f["best_valid_loss"] for f in m["folds"]] for m in results["outputs"]])
        mean_losses = np.mean(losses, axis=1)
        std_losses = np.std(losses, axis=1) / np.sqrt(self.k_fold - 1)

        f, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        axs[0].errorbar(
            x=np.arange(len(hparams)),
            y=np.mean(xy[sorted_idx, :, 0], axis=1),
            yerr=np.std(xy[sorted_idx, :, 0], axis=1) / np.sqrt(self.k_fold - 1),
            fmt="-",
            capsize=3,
            label="$\sum \widehat{x_i}$",
        )
        axs[0].errorbar(
            x=np.arange(len(hparams)),
            y=np.mean(xy[sorted_idx, :, 1], axis=1),
            yerr=np.std(xy[sorted_idx, :, 1], axis=1) / np.sqrt(self.k_fold - 1),
            fmt="-",
            capsize=3,
            label="$\sum \widehat{y_i}$",
        )
        axs[1].errorbar(
            x=np.arange(len(hparams)),
            y=mean_losses[sorted_idx],
            yerr=std_losses[sorted_idx],
            capsize=3,
            fmt="-",
            label="Erro Quadrático Médio",
        )
        axs[1].set_xticks(
            np.arange(len(hparams)),
            [tuple(d.values()) for d in hparams.to_dict(orient="records")],
            rotation="vertical",
        )

        axs[0].legend()
        axs[0].axhline(1.0, ls="--")
        axs[0].grid(True)
        axs[1].legend(loc="lower right")
        axs[1].grid(True)

        for i in range(int(np.ceil(9 / 2))):
            axs[0].axvspan(-0.5 + 6 * i, 2.5 + 6 * i, alpha=0.2)
            axs[1].axvspan(-0.5 + 6 * i, 2.5 + 6 * i, alpha=0.2)
        plt.subplots_adjust(hspace=0.1)
        f.tight_layout()
        f.savefig(os.path.join(viz_folder, "mse_with_soft_constraint_plot.png"), dpi=600)
        return results
