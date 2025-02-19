import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from datetime import datetime
from neqsim.thermo import TPflash
from src.data.handlers import DataLoader
from src.models.regression.train_models import RegressionTraining
from src.models.regression.evaluate_models import RegressionAnalysis
from src.utils import create_fluid
from src.utils.constants import P_MIN_MAX, T_MIN_MAX, FEATURES_NAMES, REGRESSION_TARGET_NAMES, K_FOLDS
from src.visualization.styles.formatting import errorbar_kwargs
from typing import List, Tuple

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

DECIMALS = 3
DPI = 400


class RegressionViz:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.k_folds = K_FOLDS
        training = RegressionTraining()
        analysis = RegressionAnalysis()
        self.data_loader = DataLoader()

        cv_data, _ = self.data_loader.load_cross_validation_datasets(problem="regression")
        self.valid_data = cv_data["valid"]
        self.results = training.load_training_models()
        self.indices = analysis.load_performance_indices()
        self.viz_folder = os.path.join("data", "models", "regression", "saved_viz")

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
        nn_models = [o["folds"][0]["model"] for o in self.results["outputs"] if o["model_type"] == "neural_network"]
        num_params = [np.sum([np.prod(var.shape.as_list()) for var in m.trainable_variables]) for m in nn_models]

        model_names = [res["model_name"].replace("#", "\#") for res in self.results["outputs"]]
        data = {}
        for name in self.indices.keys():
            index = self.indices[name]
            mean, std = index.mean(axis=(0, 2)), index.mean(axis=2).std(axis=0) / np.sqrt(self.k_folds - 1)
            data[name.replace("_", "\_")] = [rf"{mu:.2f} \textpm {sigma:.2f}" for mu, sigma in zip(mean, std)]
        data["num_params"] = num_params

        table = pd.DataFrame(data, index=model_names)

        def highlight(s, props=""):
            if s.name == "num_params":
                return np.where(s == np.min(s.values), props, "")
            else:
                mu = s.apply(lambda row: float(row.split(r" \textpm ")[0]))
                return np.where(mu == np.min(mu.values), props, "")

        table = table.style.apply(highlight, props="font-weight:bold;", axis=0)
        table.to_latex(
            os.path.join(self.viz_folder, "performance_indices_table.tex"),
            hrules=True,
            convert_css=True,
            column_format="lccccc",
        )

    def errorbar_plot(self, indices_names: List[str], by_model: bool):
        outputs = self.results["outputs"]
        labels = [hp["model_name"].replace("#", "\#") for hp in outputs]
        x = np.array([i + 1 for i in np.arange(len(outputs))])

        if by_model:
            f1, axs1 = plt.subplots(len(indices_names), 1, figsize=(6, 5 * len(indices_names)), sharex=True)
            for i, name in enumerate(indices_names):
                label = "Erro Médio Absoluto" if name == "mean_absolute_error" else "Erro Médio Quadrático"
                ax1 = axs1[i] if len(indices_names) > 1 else axs1
                y = self.indices[name].mean(axis=(0, 2))
                y_err = self.indices[name].mean(axis=2).std(axis=0) / np.sqrt(self.k_folds - 1)
                ax1.errorbar(x, y, y_err, c=f"C{i}", label=label, **errorbar_kwargs)
                ax1.grid()
                ax1.set_xticks(x, labels, rotation=90, ha="center")
                ax1.legend()
            f1.tight_layout()
            f1.savefig(os.path.join(self.viz_folder, "errorbar_plot_by_model.png"), dpi=DPI)
        else:
            indices_names = ["mean_absolute_error"]
            x = np.array([i + 1 for i in np.arange(len(REGRESSION_TARGET_NAMES))])

            for j, output in enumerate(outputs):
                model_id = output["model_id"]
                f, axs = plt.subplots(len(indices_names), 1, figsize=(6, 4 * len(indices_names)), sharex=True)
                for i, name in enumerate(indices_names):
                    label = "Erro Médio Absoluto" if name == "mean_absolute_error" else "Erro Médio Quadrático"
                    kwargs = {"label": label, **errorbar_kwargs}
                    ax = axs[i] if len(indices_names) > 1 else axs
                    y = self.indices[name].mean(axis=0)[j, :]
                    y_err = self.indices[name].std(axis=0)[j, :] / np.sqrt(self.k_folds - 1)
                    ax.errorbar(x, y, y_err, c=f"C{i}", **kwargs)
                    ax.grid()

                    def fix_ticks1(name):
                        var, component = name.split("_")
                        return "$" + var + "_{" + component + "}$"

                    def fix_ticks2(name):
                        char1, char2 = name[0], name[-1]
                        return "$" + char1 + "_{" + char2 + "}$"

                    ticks1 = [fix_ticks1(name) for name in REGRESSION_TARGET_NAMES[:-1]]
                    ticks2 = [fix_ticks2(name) for name in REGRESSION_TARGET_NAMES[-1:]]
                    ax.set_xticks(x, ticks1 + ticks2, rotation=90, ha="center")
                    ax.legend()
                f.tight_layout()
                f.savefig(os.path.join(self.viz_folder, f"errorbar_plot_by_target_model_id={model_id}.png"), dpi=DPI)

    def create(self):
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot(indices_names=["mean_absolute_error"], by_model=False)
        self.errorbar_plot(indices_names=["mean_absolute_error"], by_model=True)
