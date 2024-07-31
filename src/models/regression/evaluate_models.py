import numpy as np
import os
import tensorflow as tf

from src.data.handlers import DataLoader
from src.models.regression.train_models import RegressionTraining


class RegressionAnalysis:
    def __init__(self, samples_per_composition: int):
        self.results_folder = os.path.join(
            "src",
            "models",
            "regression",
            "saved_performance_indices",
            f"{samples_per_composition:03d}points",
        )

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        data_loader = DataLoader()
        datasets, min_max = data_loader.load_cross_validation_datasets(
            problem="regression",
            samples_per_composition=samples_per_composition,
        )

        self.valid_datasets = datasets["valid"]
        self.training = RegressionTraining(samples_per_composition=samples_per_composition)
        self.results = self.training.load_training_models()

        num_folds = len(self.valid_datasets)
        num_models = len(self.results["outputs"])
        output_size = 25

        # Initialize performance indices variables
        self.mse = np.zeros((num_folds, num_models, output_size))
        self.mae = np.zeros((num_folds, num_models, output_size))

    def run(self):
        for i, models in enumerate(self.results["outputs"]):
            for j, (result, valid_data) in enumerate(zip(models["folds"], self.valid_datasets)):
                valid_features, valid_labels = valid_data["features"], valid_data["targets"]

                X_valid = tf.convert_to_tensor(valid_features)
                Y_valid = tf.convert_to_tensor(valid_labels)

                model = result["model"]
                Y_hat_valid = model(X_valid)

                self.mae[j, i] = self.mean_absolute_error(Y_valid, Y_hat_valid)
                self.mse[j, i] = self.mean_squared_error(Y_valid, Y_hat_valid)

        self.__save_performance_indices()

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.abs(y_true - y_pred).mean(axis=0)

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.square(y_true - y_pred).mean(axis=0)

    def get_performance_indices(self):
        return {"mean_absolute_error": self.mae, "mean_squared_error": self.mse}

    def __save_performance_indices(self):
        np.savez(os.path.join(self.results_folder, "indices.npz"), **self.get_performance_indices())

    def load_performance_indices(self):
        return {**np.load(os.path.join(self.results_folder, "indices.npz"))}
