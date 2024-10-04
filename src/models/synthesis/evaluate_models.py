import numpy as np
import os
import tensorflow as tf
import umap

from scipy import stats
from scipy.special import rel_entr
from src.data.handlers import DataLoader
from src.models.synthesis.train_models import SynthesisTraining


class SynthesisAnalysis:
    def __init__(self):
        self.results_folder = os.path.join("data", "models", "synthesis", "saved_performance_indices")

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        data_loader = DataLoader()
        datasets, min_max = data_loader.load_cross_validation_datasets(problem="synthesis")

        self.min_max = min_max
        self.valid_datasets = datasets["valid"]
        self.training = SynthesisTraining()
        self.results = self.training.load_training_models()

        self.umap_model = umap.UMAP(n_neighbors=50, n_components=2, metric="euclidean", min_dist=0.1)

        self.num_folds = len(self.valid_datasets)
        self.num_models = len(self.results["outputs"])
        self.bins = 15

        # Initialize performance indices variables
        self.kldiv = np.zeros((self.num_folds, self.num_models))

    def run(self):
        for fold in range(self.num_folds):
            data = self.valid_datasets[fold]["features"].values
            size = data.shape[0]

            for model in self.results["outputs"]:
                model_fold = model["folds"][fold]

                if model["model_type"] == "dirichlet":
                    alpha = model_fold["alpha"]
                    gen_data = stats.dirichlet.rvs(alpha, size=size)

                elif model["model_type"] == "wgan":
                    generator = model_fold["generator"]
                    latent_dim = model["params"]["latent_dim"]
                    noise = tf.random.normal([size, latent_dim])
                    gen_data = generator(noise, training=False)

                data = np.r_[data, gen_data]

            # Reduce dimensions using UMAP algorithm
            reduced_data = self.umap_model.fit_transform(data)
            reduced_datasets = np.split(reduced_data, self.num_models + 1, axis=0)
            ground_truth_data = reduced_datasets.pop(0)

            # Obtain 2D histogram for datasets
            H_truth, xedges, yedges, xdelta, ydelta = self.normalized_histogram(ground_truth_data, bins=self.bins)

            for i, data in enumerate(reduced_datasets):
                H_model, _, _, _, _ = self.normalized_histogram(data, bins=[xedges, yedges])
                dkl = self.kl_divergence(H_truth, H_model).sum()
                self.kldiv[fold, i] = dkl

        self.__save_performance_indices()

    def normalized_histogram(self, data, bins):
        H, xedges, yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=bins,
            density=True,
        )
        xdelta = np.asarray([xedges[i + 1] - xedges[i] for i in range(H.shape[0])]).reshape(1, -1)
        ydelta = np.asarray([yedges[i + 1] - yedges[i] for i in range(H.shape[1])]).reshape(-1, 1)
        return H * xdelta * ydelta, xedges, yedges, xdelta, ydelta

    def kl_divergence(self, p: np.ndarray, q: np.ndarray):
        p_mod, q_mod = p.copy(), q.copy()
        q_mod = np.where((q_mod == 0) & (p_mod > 0), 1e-6, q_mod)
        return rel_entr(p_mod, q_mod)

    def get_performance_indices(self):
        return {"kl_divergence": self.kldiv}

    def __save_performance_indices(self):
        np.savez(os.path.join(self.results_folder, "indices.npz"), **self.get_performance_indices())

    def load_performance_indices(self):
        return {**np.load(os.path.join(self.results_folder, "indices.npz"))}
