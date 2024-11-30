import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from datetime import datetime
from dirichlet import mle
from glob import glob
from src.data.handlers import DataLoader
from src.models.synthesis.experiments import hparams
from src.models.synthesis.wgan import WGANGP, MLPCritic, MLPGenerator, critic_loss, generator_loss, CustomHistory
from src.utils import load_model_hparams
from src.utils.constants import K_FOLDS
from tqdm import tqdm


class SynthesisTraining:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.n_folds = K_FOLDS
        self.results_folder = os.path.join("data", "models", "synthesis", "saved_models")

    def run(self):
        data_loader = DataLoader()
        cv_data, _ = data_loader.load_cross_validation_datasets(problem="synthesis")
        train_data = cv_data["train"]

        results = {"outputs": []}
        training_start = datetime.now()
        for hp in load_model_hparams(hparams):
            model_name = hp["model_name"]
            model_type = hp["model_type"]

            model_results = {**hp}
            folds = []

            print(f"\nModel: {model_name}")
            self.logger.info(f"Model: {model_name}")

            training_model_start = datetime.now()
            pbar = tqdm(total=len(train_data))
            for fold, train in enumerate(train_data):
                pbar.set_description(f"Train using fold {fold+1} dataset")

                train_features = train["features"]

                if model_type == "dirichlet":
                    alpha = mle(train_features.values)
                    folds.append({"fold": fold + 1, "alpha": alpha})
                    self.logger.info(f"Fold {fold+1} dataset, first 3 alphas: {alpha[:3]}")
                elif model_type == "wgan":
                    params = hp["params"]
                    optim_params = hp["opt"]

                    latent_dim = params["latent_dim"]
                    lambda_ = optim_params["lambda"]
                    n_critic = optim_params["n_critic"]
                    lr = optim_params["lr"]
                    beta_1, beta_2 = optim_params["beta_1"], optim_params["beta_2"]
                    batch_size = optim_params["batch_size"]
                    epochs = optim_params["epochs"]

                    output_dim = train_features.shape[-1]

                    data = train_features.values.astype(np.float32)
                    dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

                    critic = MLPCritic(hidden_units=params["hidden_units"])
                    generator = MLPGenerator(output_dim=output_dim, hidden_units=params["hidden_units"])

                    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
                    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)

                    # Get the Wasserstein GAN model
                    wgan = WGANGP(
                        critic=critic,
                        generator=generator,
                        latent_dim=latent_dim,
                        n_critic=n_critic,
                        lambda_=lambda_,
                    )

                    # Compile the Wasserstein GAN model
                    wgan.compile(
                        critic_optimizer=critic_optimizer,
                        generator_optimizer=generator_optimizer,
                        critic_loss_fn=critic_loss,
                        generator_loss_fn=generator_loss,
                    )

                    callbacks = [CustomHistory()]
                    wgan.fit(dataset, epochs=epochs, callbacks=callbacks, verbose=0)

                    neg_critic_loss = -np.array(callbacks[0].history["critic_loss"])
                    self.logger.info(f"{params}, fold: {fold}, critic loss: {np.mean(neg_critic_loss[-10:])}")
                    folds.append(
                        {
                            "fold": fold + 1,
                            "critic": critic,
                            "generator": generator,
                            "neg_critic_loss": neg_critic_loss,
                        }
                    )

                pbar.update()

            pbar.close()
            os.system("cls" if os.name == "nt" else "clear")
            model_results["folds"] = folds
            results["outputs"].append(model_results)
            self.logger.info(f"{model_name} training elapsed Time: {datetime.now() - training_model_start}")

        self.logger.info("Saving models")
        self.logger.info(f"Total elapsed Time: {datetime.now() - training_start}")
        self.save_training_models(results)

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
                        "params": {"hidden_units": List[int], "activation": str},
                        "opt": {"lr": float, "epochs": int, "batch_size": int},
                        "folds": [
                            {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
                            {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
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

                fold_folder = os.path.join(model_folder, f"Fold{fold}")
                if not os.path.isdir(fold_folder):
                    os.mkdir(fold_folder)

                if output["model_type"] == "wgan":
                    critic = fold_results["critic"]
                    generator = fold_results["generator"]
                    neg_critic_loss = fold_results["neg_critic_loss"]

                    # Saving critic loss
                    self.save_pickle(os.path.join(fold_folder, "neg_critic_loss.pickle"), neg_critic_loss)

                    # Saving tf.keras.Model weights
                    critic.save(os.path.join(fold_folder, "critic.keras"))
                    generator.save(os.path.join(fold_folder, "generator.keras"))
                elif output["model_type"] == "dirichlet":
                    alpha = fold_results["alpha"]
                    self.save_pickle(os.path.join(fold_folder, "alpha.pickle"), alpha)

    def load_training_models(self):
        """Load synthesis models training results

        Parameters
        ----------
        results : dict
            Model training results structure. Format below:
            {
                "outputs": [
                    {
                        "model_id": int,
                        "model_name": str,
                        "params": {"hidden_units": List[int], "activation": str},
                        "opt": {"lr": float, "epochs": int, "batch_size": int},
                        "folds": [
                            {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
                            {"fold": int, "model": tf.keras.Model, "history": tf.keras.callbacks.History},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        results = {}

        model_results = []
        for folder in glob(os.path.join(self.results_folder, "*")):
            folds = []
            model_obj = self.load_pickle(os.path.join(folder, "model_info.pickle"))

            if model_obj["model_type"] == "wgan":
                for fold in np.arange(self.n_folds):
                    # critic = tf.keras.models.load_model(os.path.join(folder, f"Fold{fold+1}", "critic.keras"))
                    generator = tf.keras.models.load_model(os.path.join(folder, f"Fold{fold+1}", "generator.keras"))
                    neg_critic_loss = self.load_pickle(os.path.join(folder, f"Fold{fold+1}", "neg_critic_loss.pickle"))
                    folds.append(
                        {
                            "fold": fold + 1,
                            # "critic": critic,
                            "generator": generator,
                            "neg_critic_loss": neg_critic_loss,
                        }
                    )
            elif model_obj["model_type"] == "dirichlet":
                for fold in np.arange(self.n_folds):
                    alpha = self.load_pickle(os.path.join(folder, f"Fold{fold+1}", "alpha.pickle"))
                    folds.append({"fold": fold + 1, "alpha": alpha})

            model_results.append({"folds": folds, **model_obj})

        results["outputs"] = sorted(model_results, key=lambda item: item["model_id"])
        return results
