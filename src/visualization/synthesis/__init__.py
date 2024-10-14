import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.data.handlers import DataLoader
from src.models.synthesis.train_models import SynthesisTraining
from src.models.synthesis.evaluate_models import SynthesisAnalysis


DPI = 400


class SynthesisViz:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.k_folds = 5
        training = SynthesisTraining()
        analysis = SynthesisAnalysis()
        data_loader = DataLoader()

        cv_data, _ = data_loader.load_cross_validation_datasets(problem="synthesis")
        self.valid_data = cv_data["valid"]
        self.results = training.load_training_models()
        self.indices = analysis.load_performance_indices()
        self.viz_folder = os.path.join("data", "visualization", "synthesis", "saved_viz")

        if not os.path.isdir(self.viz_folder):
            os.makedirs(self.viz_folder)

    def models_table(self):
        outputs = self.results["outputs"]
        outputs = [
            {"model_id": model["model_id"], "model_name": model["model_name"], **model["params"], **model["opt"]}
            for model in outputs
        ]

        table = pd.DataFrame.from_records(outputs)
        table.to_latex(os.path.join(self.viz_folder, "models_table.tex"), index=False)

    def performance_indices_table(self):
        pass

    def errorbar_plot(self):
        kl_divergence = self.indices["kl_divergence"]
        wasserstein_dist = self.indices["wasserstein_distance"]
        models = [model["model_name"].replace("#", "\#") for model in self.results["outputs"]][1:]

        x = np.arange(len(models))
        y = np.mean(kl_divergence, axis=0)
        yerr = np.std(kl_divergence, axis=0) / np.sqrt(self.n_folds - 1)

        f, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        axs[0].errorbar(x=x, y=y[1:], yerr=yerr[1:], fmt="", capsize=3)
        axs[0].grid(True)
        # axs[0].set_ylabel(r"$\widehat{D}_{\mathrm{KL}}(P|Q)$")
        axs[0].set_ylabel("KL Divergence")

        y = np.mean(wasserstein_dist, axis=0)
        yerr = np.std(wasserstein_dist, axis=0) / np.sqrt(self.n_folds - 1)

        axs[1].errorbar(x=x, y=y[1:], yerr=yerr[1:], fmt="", capsize=3)
        axs[1].grid(True)
        # axs[1].set_ylabel(r"$\widehat{W}_{1}(\mu,\nu)$")
        axs[1].set_ylabel("Wasserstein Distance")
        axs[1].set_xticks(x, models, rotation=90, ha="center")

        f.subplots_adjust(hspace=0.1)
        f.savefig(os.path.join(self.viz_folder, "errorbar_plot.png"), dpi=DPI)

    def create(self):
        pass
