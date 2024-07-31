import numpy as np
import os
import tensorflow as tf

from scipy.stats import gmean
from sklearn.metrics import confusion_matrix
from src.data.handlers import DataLoader
from src.models.classification.train_models import ClassificationTraining


class ClassificationAnalysis:
    def __init__(self, samples_per_composition: int):
        self.results_folder = os.path.join(
            "src",
            "models",
            "classification",
            "saved_performance_indices",
            f"{samples_per_composition:03d}points",
        )

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        data_loader = DataLoader()
        datasets, _ = data_loader.load_cross_validation_datasets(
            problem="classification",
            samples_per_composition=samples_per_composition,
        )

        self.valid_datasets = datasets["valid"]
        self.training = ClassificationTraining(samples_per_composition=samples_per_composition)
        self.results = self.training.load_training_models()

        num_classes = 3
        num_folds = len(self.valid_datasets)
        num_models = len(self.results["outputs"])

        # Initialize performance indices variables
        self.confusion_matrices = np.zeros((num_folds, num_models, num_classes, num_classes))
        self.accuracies = np.zeros((num_folds, num_models))
        self.sensitivities = np.zeros((num_folds, num_models, num_classes))
        self.sp_indexes = np.zeros((num_folds, num_models))
        self.cross_entropy = np.zeros((num_folds, num_models))
        self.bic = np.zeros((num_folds, num_models))
        self.aic = np.zeros((num_folds, num_models))

    def run(self):
        # Confusion Matrix, Cross-Entropy
        for i, models in enumerate(self.results["outputs"]):
            for j, (result, valid_data) in enumerate(zip(models["folds"], self.valid_datasets)):
                valid_features, valid_labels = valid_data["features"], valid_data["targets"]

                X_valid = tf.convert_to_tensor(valid_features)
                probs = tf.convert_to_tensor(valid_labels)
                y_valid = tf.argmax(probs, axis=1)

                model = result["model"]
                logits = model(X_valid)
                probs_hat = tf.nn.softmax(logits)
                y_valid_hat = tf.argmax(probs_hat, axis=1)
                self.confusion_matrices[j, i] = confusion_matrix(y_valid, y_valid_hat)

                # Cross-Entropy
                cross_entropy_values = tf.keras.losses.categorical_crossentropy(probs, probs_hat)
                self.cross_entropy[j, i] = tf.reduce_mean(cross_entropy_values)

        # Accuracy, Sensitivity, Sum-Product Index
        for model in np.arange(self.confusion_matrices.shape[1]):
            for fold in np.arange(self.confusion_matrices.shape[0]):
                cm = self.confusion_matrices[fold, model, :, :]

                # Accuracy
                self.accuracies[fold, model] = np.diag(cm).sum() / cm.sum()

                # Sensitivity
                sens = np.diag(cm) / cm.sum(axis=1)
                self.sensitivities[fold, model] = sens

                # SP Index
                self.sp_indexes[fold, model] = np.sqrt(np.mean(sens) * gmean(sens))

        self.__save_performance_indices()

    def get_performance_indices(self):
        return {
            "accuracy": self.accuracies,
            "confusion_matrix": self.confusion_matrices,
            "cross_entropy": self.cross_entropy,
            "sensitivity": self.sensitivities,
            "sp_index": self.sp_indexes,
        }

    def __save_performance_indices(self):
        np.savez(os.path.join(self.results_folder, "indices.npz"), **self.get_performance_indices())

    def load_performance_indices(self):
        return {**np.load(os.path.join(self.results_folder, "indices.npz"))}
