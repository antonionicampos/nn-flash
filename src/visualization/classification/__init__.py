import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.metrics import RocCurveDisplay, auc
from src.data.handlers import DataLoader
from src.models.classification.train_models import Training
from src.models.classification.evaluate_models import Analysis
from src.models.classification.utils import binary_classification
from src.utils.constants import TARGET_NAMES
from typing import List, Dict, Tuple, Any

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

DECIMALS = 3
DPI = 400


class Viz:

    def __init__(self, samples_per_composition: int):
        training = Training(samples_per_composition=samples_per_composition)
        analysis = Analysis(samples_per_composition=samples_per_composition)
        data_loader = DataLoader(problem="classification", samples_per_composition=samples_per_composition)

        cv_data = data_loader.load_cross_validation_datasets()
        self.valid_data = cv_data["valid"]
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

    def roc_analysis(
        self,
        model_id: int,
        figsize: Tuple[int] = (15, 5),
        xlim: Tuple[float] = (),
        ylim: Tuple[float] = (),
    ):
        model_info = [info for info in self.results["outputs"] if info["model_id"] == model_id][0]
        n_folds = len(model_info["folds"])

        f, axs = plt.subplots(1, 3, figsize=figsize)
        for label in range(3):
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)

            for fold, data in enumerate(model_info["folds"]):
                model = data["model"]
                y, y_hat = binary_classification(model, self.valid_data[fold], label=label)

                viz = RocCurveDisplay.from_predictions(
                    y,
                    y_hat,
                    name=f"ROC fold {fold + 1}",
                    alpha=0.4,
                    lw=1,
                    ax=axs[label],
                    plot_chance_level=(fold == n_folds - 1),
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            axs[label].step(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            axs[label].fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.4,
                label=r"$\pm 1 \sigma$",
            )

            axs[label].set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title=f"Mean ROC curve with uncertainty\n(Positive label '{TARGET_NAMES[label]}')",
            )
            axs[label].legend(loc="lower right", prop={"size": 8})
            axs[label].grid()

            if xlim:
                axs[label].set_xlim(xlim)
            if ylim:
                axs[label].set_ylim(ylim)
        f.tight_layout()
        f.savefig(os.path.join(self.viz_folder, f"roc_analysis_model_id={model_id}.png"), dpi=DPI)

    def create(self):
        for model_info in self.results["outputs"]:
            self.roc_analysis(model_id=model_info["model_id"])
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot(indices_names=["sp_index", "cross_entropy"])
