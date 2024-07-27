import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.models.classification.train_models import Training
from src.models.classification.evaluate_models import Analysis
from src.utils.constants import TARGET_NAMES
from typing import List

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

DECIMALS = 3
DPI = 400


class Viz:

    def __init__(self, samples_per_composition: int):
        training = Training(samples_per_composition=samples_per_composition)
        analysis = Analysis(samples_per_composition=samples_per_composition)

        self.results = training.load_training_models(samples_per_composition=samples_per_composition)
        self.indices = analysis.load_performance_indices()
        self.viz_folder = os.path.join(
            "src",
            "visualization",
            "classification",
            "saved_viz",
            f"{samples_per_composition:03d}points",
        )

        if not os.path.isdir(self.viz_folder):
            os.makedirs(self.viz_folder)

    def models_table(self):
        outputs = self.results["outputs"]
        outputs = [
            {"model_id": model["model_id"], "model_name": model["model_name"], **model["arch"], **model["opt"]}
            for model in outputs
        ]

        table = pd.DataFrame.from_records(outputs)
        table.to_latex(os.path.join(self.viz_folder, "models_table.tex"), index=False)

    def performance_indices_table(self):
        model_names = [res["model_name"] for res in self.results["outputs"]]

        data = {}
        for name in self.indices.keys():
            index = self.indices[name]
            mean = np.round(index.mean(axis=0), DECIMALS)
            std = np.round(index.std(axis=0), DECIMALS)
            if len(self.indices[name].shape) == 2:
                data[name] = [f"{mu} +/- {sigma}" for mu, sigma in zip(mean, std)]
            elif name == "sensitivity":
                for i, label in enumerate(TARGET_NAMES):
                    lab = label.lower()
                    data[f"{name}_{lab}"] = [f"{mu} +/- {sigma}" for mu, sigma in zip(mean[:, i], std[:, i])]

        table = pd.DataFrame(data, index=model_names)
        table.to_latex(os.path.join(self.viz_folder, "performance_indices_table.tex"))

    def errorbar_plot(self, indices_names: List[str]):
        outputs = self.results["outputs"]
        labels = [hp["model_name"].replace("#", "\#") for hp in outputs]
        x = np.array([i + 1 for i in np.arange(len(outputs))])

        f, axs = plt.subplots(len(indices_names), 1, figsize=(5, 4 * len(indices_names)), sharex=True)

        for i, name in enumerate(indices_names):
            ax = axs[i] if len(indices_names) > 1 else axs
            y, y_err = self.indices[name].mean(axis=0), self.indices[name].std(axis=0)
            ax.errorbar(x, y, y_err, c=f"C{i}", fmt="o", elinewidth=2.0, capsize=3.0, capthick=2.0, label=name)
            ax.yaxis.grid()
            ax.set_xticks(x, labels, rotation=90, ha="center")
            ax.legend()

        f.tight_layout()
        f.savefig(os.path.join(self.viz_folder, "errorbar_plot.png"), dpi=DPI)

    def create(self):
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot(indices_names=["sp_index", "cross_entropy"])
