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
from src.utils.constants import TARGET_NAMES, P_MIN_MAX, T_MIN_MAX, FEATURES_NAMES, REGRESSION_TARGET_NAMES
from typing import List, Tuple

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

DECIMALS = 3
DPI = 400


class RegressionViz:

    def __init__(self, samples_per_composition: int):
        self.logger = logging.getLogger(__name__)
        self.k_folds = 10
        training = RegressionTraining(samples_per_composition=samples_per_composition)
        analysis = RegressionAnalysis(samples_per_composition=samples_per_composition)
        self.data_loader = DataLoader()

        cv_data, _ = self.data_loader.load_cross_validation_datasets(
            problem="regression",
            samples_per_composition=samples_per_composition,
        )
        self.valid_data = cv_data["valid"]
        self.results = training.load_training_models()
        self.indices = analysis.load_performance_indices()
        self.viz_folder = os.path.join(
            "data",
            "visualization",
            "regression",
            "saved_viz",
            f"{samples_per_composition:03d}points",
        )

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
        model_names = [res["model_name"].replace("#", "\#") for res in self.results["outputs"]]
        data = {}
        for name in self.indices.keys():
            index = self.indices[name]
            mean, std = index.mean(axis=(0, 2)), index.std(axis=(0, 2))
            # if len(self.indices[name].shape) == 3:
            #     data[name] = [f"{mu} +/- {sigma}" for mu, sigma in zip(mean, std)]
            data[name.replace("_", "\_")] = [rf"{mu:.3f} \textpm {sigma:.3f}" for mu, sigma in zip(mean, std)]

        table = pd.DataFrame(data, index=model_names)

        def highlight(s, props=""):
            mu = s.apply(lambda row: float(row.split(r" \textpm ")[0]))
            return np.where(mu == np.min(mu.values), props, "")

        table = table.style.apply(highlight, props="font-weight:bold;", axis=0)
        table.to_latex(os.path.join(self.viz_folder, "performance_indices_table.tex"), hrules=True, convert_css=True)
        # table.to_latex(os.path.join(self.viz_folder, "performance_indices_table.tex"))

    def errorbar_plot(self, indices_names: List[str], by_model: bool):
        outputs = self.results["outputs"]
        labels = [hp["model_name"].replace("#", "\#") for hp in outputs]
        x = np.array([i + 1 for i in np.arange(len(outputs))])

        errorbar_kwargs = {"fmt": "_", "ms": 4.0, "mew": 1.0, "elinewidth": 1.0, "capsize": 2.0, "capthick": 1.0}
        if by_model:
            f, axs = plt.subplots(len(indices_names), 1, figsize=(6, 5 * len(indices_names)), sharex=True)
            for i, name in enumerate(indices_names):
                ax = axs[i] if len(indices_names) > 1 else axs
                y = self.indices[name].mean(axis=(0, 2))
                y_err = self.indices[name].std(axis=(0, 2)) / np.sqrt(self.k_folds)
                ax.errorbar(x, y, y_err, c=f"C{i}", label=name, **errorbar_kwargs)
                ax.yaxis.grid()
                ax.set_xticks(x, labels, rotation=90, ha="center")
                ax.legend()
            f.tight_layout()
            f.savefig(os.path.join(self.viz_folder, "errorbar_plot_by_model.png"), dpi=DPI)
        else:
            indices_names = ["mean_absolute_error"]
            x = np.array([i + 1 for i in np.arange(len(REGRESSION_TARGET_NAMES))])

            for j, output in enumerate(outputs):
                model_id = output["model_id"]
                f, axs = plt.subplots(len(indices_names), 1, figsize=(6, 4 * len(indices_names)), sharex=True)
                for i, name in enumerate(indices_names):
                    kwargs = {"label": name, **errorbar_kwargs}
                    ax = axs[i] if len(indices_names) > 1 else axs
                    y = self.indices[name].mean(axis=0)[j, :]
                    y_err = self.indices[name].std(axis=0)[j, :] / np.sqrt(self.k_folds)
                    ax.errorbar(x, y, y_err, c=f"C{i}", **kwargs)
                    ax.yaxis.grid()

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

    def phase_diagram(
        self,
        model_ids: List[int],
        samples_per_composition: int,
        label: int = 1,
        fold: int = 0,
        use_mean_prediction=False,
        xlim: Tuple[float] = (),
        ylim: Tuple[float] = (),
    ):
        self.logger.info("Starting phase diagram generation")
        datasets = self.data_loader.load_cross_validation_datasets(
            problem="classification",
            samples_per_composition=samples_per_composition,
        )

        data = pd.concat(datasets["test"], axis=0, ignore_index=True)

        # generate phase diagram data
        size = 400

        # Data preparation
        pressure = np.linspace(*P_MIN_MAX, num=size)
        temperature = np.linspace(*T_MIN_MAX, num=size)

        PP, TT = np.meshgrid(pressure, temperature)
        P = PP.flatten().reshape(-1, 1)
        T = TT.flatten().reshape(-1, 1)

        sample = np.random.randint(data.shape[0])
        data = data.iloc[sample : sample + 1, :].copy()
        data = pd.DataFrame(np.tile(data.values, (size * size, 1)), columns=data.columns)
        data[["P", "T"]] = np.c_[P, T]
        self.logger.info(f"Sample size: {data.shape[0]}")

        # Preprocessing
        data[FEATURES_NAMES[:-2]] = data[FEATURES_NAMES[:-2]] / 100.0

        P_min, P_max = P_MIN_MAX
        T_min, T_max = T_MIN_MAX
        data["P"] = (data["P"] - P_min) / (P_max - P_min)
        data["T"] = (data["T"] - T_min) / (T_max - T_min)

        features = data[FEATURES_NAMES].copy()
        features = features.apply(lambda s: pd.to_numeric(s))

        # Fluid creation
        composition = data.iloc[0, data.columns.str.startswith("z")].to_dict()
        fluid = create_fluid(composition)

        models_info = list(filter(lambda item: item["model_id"] in model_ids, self.results["outputs"]))
        for model_info in models_info:
            model_id = model_info["model_id"]
            model_name = model_info["model_name"]
            labels = []
            # Call simulation
            start = datetime.now()
            for d in data.to_dict(orient="records"):
                fluid.setPressure(d["P"], "bara")
                fluid.setTemperature(d["T"], "K")

                TPflash(fluid)

                phases = [p for p in fluid.getPhases() if p]
                label_name = ",".join([str(phase.getPhaseTypeName()) for phase in phases])

                if label_name == "oil,liquid":
                    labels.append("oil")
                elif label_name == "gas,liquid":
                    labels.append("gas")
                elif label_name == "gas,oil":
                    labels.append("mix")

            labels = pd.get_dummies(pd.Series(labels)).astype(int).values

            self.logger.info(f"Model ID: {model_id}, model name: {model_name}")
            self.logger.info(f"Flash simulation Elapsed Time: {datetime.now() - start}")

            # Call neural network model predictions
            if use_mean_prediction:  # Mean and Standard Dev for all folds
                ps = []
                start = datetime.now()
                for fold_info in model_info["folds"]:
                    model = fold_info["model"]
                    logits = model(features.values)
                    ps.append(tf.nn.softmax(logits, axis=1).numpy())
                self.logger.info(f"Neural Net Elapsed Time: {datetime.now() - start}")
                mean_p = np.array(ps).mean(axis=0)
                std_p = np.array(ps).std(axis=0)

                # plot phase diagram and neural net probabilities heatmap
                self.logger.info("Creating phase diagram and neural net probabilities heatmap plot")
                f, axs = plt.subplots(1, 2, figsize=(14, 6))

                axs[0].contour(TT, PP, labels[:, label].reshape(size, size), levels=0, colors="white")
                axs[1].contour(TT, PP, labels[:, label].reshape(size, size), levels=0, colors="white")
                pcm0 = axs[0].pcolormesh(TT, PP, mean_p[:, label].reshape(size, size), cmap="plasma")
                pcm1 = axs[1].pcolormesh(TT, PP, std_p[:, label].reshape(size, size), cmap="Greys")
                f.colorbar(pcm0, ax=axs[0])
                f.colorbar(pcm1, ax=axs[1])

                if xlim:
                    axs[0].set_xlim(xlim)  # Temperature
                    axs[1].set_xlim(xlim)
                if ylim:
                    axs[0].set_ylim(ylim)  # Pressure
                    axs[1].set_ylim(ylim)

                f.suptitle(model_name.replace("#", "\#"))
                axs[0].set_xlabel("Temperatura [K]")
                axs[1].set_xlabel("Temperatura [K]")
                axs[0].set_ylabel("Pressão [bara]")
                axs[1].set_ylabel("Pressão [bara]")
                f.tight_layout()
                f.savefig(
                    os.path.join(self.viz_folder, f"phase_diagram_w_uncertainty_model_id={model_id}.png"),
                    dpi=DPI,
                )
            else:  # For specific fold
                start = datetime.now()
                model = model_info["folds"][fold]["model"]
                logits = model(features.values)
                probs = tf.nn.softmax(logits, axis=1).numpy()
                self.logger.info(f"Neural Net Elapsed Time: {datetime.now() - start}")

                # plot phase diagram and neural net probabilities heatmap
                self.logger.info("Creating phase diagram and neural net probabilities heatmap plot")
                f, ax = plt.subplots()
                c = ax.contour(TT, PP, labels[:, label].reshape(size, size), levels=0, colors="white")
                pcm = ax.pcolormesh(TT, PP, probs[:, label].reshape(size, size), vmin=0.0, vmax=1.0, cmap="plasma")
                f.colorbar(pcm, ax=ax)

                if xlim:
                    ax.set_xlim(xlim)  # Temperature
                if ylim:
                    ax.set_ylim(ylim)  # Pressure

                ax.legend(handles=[c])
                ax.set_title(model_name.replace("#", "\#"))
                ax.set_xlabel("Temperatura [K]")
                ax.set_ylabel("Pressão [bara]")
                f.tight_layout()
                f.savefig(os.path.join(self.viz_folder, f"phase_diagram_model_id={model_id}.png"), dpi=DPI)

    def create(self):
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot(indices_names=["mean_absolute_error"], by_model=False)
        self.errorbar_plot(indices_names=["mean_absolute_error"], by_model=True)
