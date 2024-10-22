import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.data.handlers import DataLoader
from src.models.synthesis.train_models import SynthesisTraining
from src.models.synthesis.evaluate_models import SynthesisAnalysis
from src.visualization.styles.formatting import errorbar_kwargs

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

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
            {
                "model_id": model["model_id"],
                "model_name": model["model_name"].replace("#", "\#"),
                **model["params"],
                # **model["opt"],
            }
            for model in outputs
            if model["model_type"] == "wgan"
        ]
        outputs.insert(0, {"model_id": self.results["outputs"][0]["model_id"], "model_name": "Dirichlet"})
        table = pd.DataFrame.from_records(outputs, index="model_id").fillna("-")
        table.to_latex(os.path.join(self.viz_folder, "models_table.tex"), index=False)

    def performance_indices_table(self):
        model_names = [res["model_name"].replace("#", "\#") for res in self.results["outputs"]]
        data = {}
        for name in self.indices.keys():
            index = self.indices[name]
            mean, std = index.mean(axis=0), index.std(axis=0) / np.sqrt(self.k_folds - 1)
            data[name] = [rf"{mu:.3f} \textpm {sigma:.3f}" for mu, sigma in zip(mean, std)]
        table = pd.DataFrame(data, index=model_names)
        table.columns = [" ".join([s.capitalize() for s in col.split("_")]) for col in table.columns]

        def highlight(s, props=""):
            mu = s.apply(lambda row: float(row.split(r" \textpm ")[0]))
            return np.where(mu == np.min(mu.values), props, "")

        table = table.style.apply(highlight, props="font-weight:bold;", axis=0)
        table.to_latex(
            os.path.join(self.viz_folder, "performance_indices_table.tex"),
            hrules=True,
            convert_css=True,
            column_format="lccccc",
        )

    def errorbar_plot(self):
        kl_divergence = self.indices["kl_divergence"]
        wasserstein_dist = self.indices["wasserstein_distance"]
        models = [model["model_name"].replace("#", "\#") for model in self.results["outputs"]]

        x = np.arange(len(models))
        y = np.mean(kl_divergence, axis=0)
        yerr = np.std(kl_divergence, axis=0) / np.sqrt(self.k_folds - 1)

        f, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        axs[0].errorbar(x=x, y=y, yerr=yerr, **errorbar_kwargs)
        axs[0].grid(True)
        axs[0].set_ylabel(r"$\widehat{D}_{\mathrm{KL}}(P|Q)$")
        # axs[0].set_ylabel("KL Divergence Estimate")

        y = np.mean(wasserstein_dist, axis=0)
        yerr = np.std(wasserstein_dist, axis=0) / np.sqrt(self.k_folds - 1)

        axs[1].errorbar(x=x, y=y, yerr=yerr, **errorbar_kwargs)
        axs[1].grid(True)
        axs[1].set_ylabel(r"$\widehat{W}_{1}(\mu,\nu)$")
        # axs[1].set_ylabel("Wasserstein Distance Estimate")
        axs[1].set_xticks(x, models, rotation=90, ha="center")

        # f.subplots_adjust(hspace=0.1)
        f.tight_layout()
        f.savefig(os.path.join(self.viz_folder, "errorbar_plot.png"), dpi=DPI)

    def create(self):
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot()
