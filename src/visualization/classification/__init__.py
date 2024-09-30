import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from datetime import datetime
from neqsim.thermo import TPflash
from sklearn.metrics import RocCurveDisplay, auc
from src.data.handlers import DataLoader
from src.models.classification.train_models import ClassificationTraining
from src.models.classification.evaluate_models import ClassificationAnalysis
from src.models.classification.utils import binary_classification
from src.utils import create_fluid
from src.utils.constants import TARGET_NAMES, P_MIN_MAX, T_MIN_MAX
from typing import List, Tuple

plt.style.use("seaborn-v0_8-paper")
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))

DPI = 600


class ClassificationViz:

    def __init__(self, samples_per_composition: int):
        self.logger = logging.getLogger(__name__)
        self.k_folds = 10
        self.samples_per_composition = samples_per_composition
        training = ClassificationTraining(samples_per_composition=samples_per_composition)
        analysis = ClassificationAnalysis(samples_per_composition=samples_per_composition)
        self.data_loader = DataLoader()

        cv_data, _ = self.data_loader.load_cross_validation_datasets(
            problem="classification",
            samples_per_composition=samples_per_composition,
        )
        self.valid_data = cv_data["valid"]
        self.logger.info("Loading training models results")
        self.results = training.load_training_models()
        self.logger.info("Loading performance indices results")
        self.indices = analysis.load_performance_indices()
        self.viz_folder = os.path.join(
            "data",
            "visualization",
            "classification",
            "saved_viz",
            f"{samples_per_composition:03d}points",
        )

        if not os.path.isdir(self.viz_folder):
            os.makedirs(self.viz_folder)

    def models_table(self):
        outputs = self.results["outputs"]

        models_tables = {}
        for model in outputs:
            if model["model_type"] == "neural_network":
                if model["model_type"] not in models_tables:
                    models_tables[model["model_type"]] = []
                models_tables[model["model_type"]].append(
                    {
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        **model["params"],
                        **model["opt"],
                    }
                )
            elif model["model_type"] == "svm":
                if model["model_type"] not in models_tables:
                    models_tables[model["model_type"]] = []
                models_tables[model["model_type"]].append(
                    {
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        **model["params"],
                    }
                )

        for k, v in models_tables.items():
            table = pd.DataFrame.from_records(v)
            table.to_latex(os.path.join(self.viz_folder, f"{k}_table.tex"), index=False)

    def performance_indices_table(self):
        model_names = [res["model_name"].replace("#", "\#") for res in self.results["outputs"]]
        data = {}
        for name in self.indices.keys():
            index = self.indices[name]
            if name == "cross_entropy":
                continue
                # mean, std = index.mean(axis=0), index.std(axis=0) / np.sqrt(self.k_folds - 1)
            else:
                mean, std = index.mean(axis=0) * 100, (index.std(axis=0) / np.sqrt(self.k_folds - 1)) * 100
            if len(self.indices[name].shape) == 2:
                data[name] = [rf"{mu:.2f} \textpm {sigma:.2f}" for mu, sigma in zip(mean, std)]
            elif name == "sensitivity":
                for i, label in enumerate(TARGET_NAMES):
                    data[f"{name}_{label.lower()}"] = [
                        rf"{mu:.2f} \textpm {sigma:.2f}" for mu, sigma in zip(mean[:, i], std[:, i])
                    ]

        table = pd.DataFrame(data, index=model_names)
        table.columns = [f"{' '.join([s.capitalize() for s in col.split('_')])} [\%]" for col in table.columns]

        def highlight(s, props=""):
            mu = s.apply(lambda row: float(row.split(r" \textpm ")[0]))
            if s.name == "cross_entropy":
                return np.where(mu == np.min(mu.values), props, "")
            else:
                return np.where(mu == np.max(mu.values), props, "")

        table = table.style.apply(highlight, props="font-weight:bold;", axis=0)
        table.to_latex(
            os.path.join(self.viz_folder, "performance_indices_table.tex"),
            hrules=True,
            convert_css=True,
            column_format="lccccc",
        )

    def errorbar_plot(self, indices_names: List[str]):
        outputs = self.results["outputs"]
        labels = [hp["model_name"].replace("#", "\#") for hp in outputs]
        x = np.array([i + 1 for i in np.arange(len(outputs))])

        for name in indices_names:
            f, ax = plt.subplots(figsize=(10, 5))

            y = self.indices[name].mean(axis=0) * 100
            y_err = (self.indices[name].std(axis=0) / np.sqrt(self.k_folds - 1)) * 100
            kwargs = {"c": "C0", "fmt": "_", "ms": 4.0, "mew": 1.0, "elinewidth": 1.0, "capsize": 2.0, "capthick": 1.0}
            ax.errorbar(x, y, y_err, label=name.replace("_", " "), **kwargs)
            ax.yaxis.grid()
            ax.set_xticks(x, labels, rotation=90, ha="center")
            ax.legend()

            f.tight_layout()
            f.savefig(os.path.join(self.viz_folder, f"{name}_errorbar_plot.png"), dpi=DPI)
            plt.close()

    def roc_curves(self, xlim: Tuple[float] = (), ylim: Tuple[float] = ()):
        n_folds = len(self.results["outputs"][0]["folds"])

        for model_info in self.results["outputs"]:
            model_type = model_info["model_type"]
            model_id = model_info["model_id"]

            self.logger.info(f"ROC curve for model ID: {model_id} and model type: {model_type}")
            f, axs = plt.subplots(1, 3, figsize=(12, 5))
            for label in range(3):
                tprs, aucs = [], []
                mean_fpr = np.linspace(0, 1, 100)

                for fold, data in enumerate(model_info["folds"]):
                    model = data["model"]

                    y, y_hat = binary_classification(model, self.valid_data[fold], label=label, model_type=model_type)

                    viz = RocCurveDisplay.from_predictions(
                        y,
                        y_hat,
                        name=f"ROC fold {fold + 1}",
                        alpha=0.4,
                        lw=1,
                        # ax=axs[label],
                        plot_chance_level=(fold == n_folds - 1),
                    )
                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs) / np.sqrt(self.k_folds - 1)
                axs[label].step(
                    mean_fpr,
                    mean_tpr,
                    color="b",
                    label=r"ROC Médio (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                    lw=1,
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
                    alpha=0.5,
                    label=r"$\pm 1 \sigma$",
                )

                axs[label].set(
                    xlabel="Taxa de Falso Positivo",
                    ylabel="Taxa de Verdadeiro Positivo",
                    title=f"Curva ROC média com incerteza\n(Rótulo positivo '{TARGET_NAMES[label]}')",
                )
                axs[label].legend(loc="lower right", prop={"size": 8})
                axs[label].grid()

                if xlim:
                    axs[label].set_xlim(xlim)
                if ylim:
                    axs[label].set_ylim(ylim)
            f.tight_layout()

            if not os.path.isdir(os.path.join(self.viz_folder, "roc_curves")):
                os.mkdir(os.path.join(self.viz_folder, "roc_curves"))
            f.savefig(os.path.join(self.viz_folder, "roc_curves", f"model_id={model_id}.png"), dpi=DPI)
            plt.close()

    def phase_diagram(
        self,
        label: int = 1,
        fold: int = 0,
        use_mean_prediction=False,
        xlim: Tuple[float] = (),
        ylim: Tuple[float] = (),
    ):
        self.logger.info("Starting phase diagram generation")
        data = pd.concat([valid["features"] for valid in self.valid_data], axis=0, ignore_index=True)

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
        features = data.copy()
        features = features.apply(lambda s: pd.to_numeric(s))
        P_min, P_max = P_MIN_MAX
        T_min, T_max = T_MIN_MAX
        features["P"] = (features["P"] - P_min) / (P_max - P_min)
        features["T"] = (features["T"] - T_min) / (T_max - T_min)

        composition = data.iloc[0, data.columns.str.startswith("z")].to_dict()
        for output in self.results["outputs"]:
            # Fluid creation
            fluid = create_fluid(composition)

            model_id = output["model_id"]
            model_name = output["model_name"]
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
                for fold_info in output["folds"]:
                    model = fold_info["model"]
                    self.logger.info(f"Fold: {fold_info['fold']}")
                    start = datetime.now()
                    logits = model(features.values)
                    ps.append(tf.nn.softmax(logits, axis=1).numpy())
                    self.logger.info(f"Neural Net Elapsed Time: {datetime.now() - start}")
                mean_p = np.array(ps).mean(axis=0)
                std_p = np.array(ps).std(axis=0) / np.sqrt(len(ps))

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
                model = output["folds"][fold]["model"]
                start = datetime.now()
                logits = model(features.values)
                probs = tf.nn.softmax(logits, axis=1).numpy()
                self.logger.info(f"Neural Net Elapsed Time: {datetime.now() - start}")

                # plot phase diagram and neural net probabilities heatmap
                self.logger.info("Creating phase diagram and neural net probabilities heatmap plot")
                f, ax = plt.subplots()
                ax.contour(TT, PP, labels[:, label].reshape(size, size), levels=0, colors="white")
                pcm = ax.pcolormesh(TT, PP, probs[:, label].reshape(size, size), vmin=0.0, vmax=1.0, cmap="plasma")
                f.colorbar(pcm, ax=ax)

                if xlim:
                    ax.set_xlim(xlim)  # Temperature
                if ylim:
                    ax.set_ylim(ylim)  # Pressure

                ax.set_title(model_name.replace("#", "\#"))
                ax.set_xlabel("Temperatura [K]")
                ax.set_ylabel("Pressão [bara]")
                f.tight_layout()
                f.savefig(os.path.join(self.viz_folder, f"phase_diagram_model_id={model_id}.png"), dpi=DPI)
            plt.close()

    def confusion_matrix_plot(self):
        model_names = [res["model_name"].replace("#", "\#") for res in self.results["outputs"]]

        for i, model_info in enumerate(self.results["outputs"]):
            model_type = model_info["model_type"]
            model_id = model_info["model_id"]

            self.logger.info(f"Confusion Matrix for model ID: {model_id} and model type: {model_type}")

            cm = self.indices["confusion_matrix"][:, i, :, :].astype("int16")

            cm = cm / np.sum(cm, axis=2)[:, :, None]

            f, ax = plt.subplots()
            cm_mean = np.mean(cm, axis=0)
            cm_std = np.std(cm, axis=0) / np.sqrt(self.k_folds - 1)

            ax.matshow(cm_mean, alpha=0.5, cmap="Greys")
            for ii in range(cm_mean.shape[0]):
                for jj in range(cm_mean.shape[1]):
                    text = f"{cm_mean[ii, jj] * 100:1.2f} \\textpm \n {cm_std[ii, jj] * 100:1.2f} \%"
                    ax.text(x=jj, y=ii, s=text, va="center", ha="center", size="x-large")

            ax.set_xlabel("Classes Estimadas")
            ax.set_ylabel("Classes Reais")
            ax.set_title(model_names[i])

            # 0: gas, 1: mix, 2: oil
            ax.set_xticks([0, 1, 2], ["Vapor", "Mistura", "Líquido"])
            ax.set_yticks([0, 1, 2], ["Vapor", "Mistura", "Líquido"])
            ax.grid(False)

            f.tight_layout()

            if not os.path.isdir(os.path.join(self.viz_folder, "confusion_matrix")):
                os.mkdir(os.path.join(self.viz_folder, "confusion_matrix"))
            plt.savefig(os.path.join(self.viz_folder, "confusion_matrix", f"model_id={model_id}.png"), dpi=DPI)
            plt.close()

    def create(self):
        self.logger.info("Starting ROC curves creation")
        self.roc_curves()
        self.confusion_matrix_plot()
        self.models_table()
        self.performance_indices_table()
        self.errorbar_plot(indices_names=["sp_index"])
        # self.phase_diagram(use_mean_prediction=False)
        # self.phase_diagram(use_mean_prediction=True)
