import glob
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from neqsim.thermo import TPflash
from neqsim.thermo.thermoTools import dataFrame, jNeqSim
from scipy.stats import dirichlet
from src.data.handlers import DataLoader as ExperimentalDataLoader
from src.models.synthesis.train_models import SynthesisTraining
from src.utils import create_fluid
from src.utils.constants import (
    NEQSIM_COMPONENTS,
    FEATURES_NAMES,
    P_MIN_MAX,
    T_MIN_MAX,
    REGRESSION_TARGET_NAMES,
    K_FOLDS,
)
from tqdm import tqdm
from typing import Any


class DataGen:

    def __init__(self, dataset_size: int):
        self.logger = logging.getLogger(__name__)
        self.random_state = 13
        self.k_folds = K_FOLDS
        self.dataset_size = dataset_size

        # P_min = 10 bara | P_max = 450 bara
        # T_min = 150 K   | T_max = 1125 K
        self.P_bounds = [10.0, 450.0]
        self.T_bounds = [150.0, 1125.0]

    def load_pickle(self, filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

    def generate_sample(self, model):
        while True:
            try:
                if "WGAN" in model["name"]:
                    x = tf.random.normal([1, model["latent_dim"]])
                    composition = model["generator"](x).numpy().flatten()
                elif "Dirichlet" in model["name"]:
                    composition = dirichlet.rvs(model["alpha"], size=1).flatten()

                composition_dict = {name: 100 * value for value, name in zip(composition, FEATURES_NAMES[:-2])}

                # Algorithm to guarantee that P and T are close to phase envelope
                # Give more information around the envelope
                fluid = create_fluid(composition_dict)
                thermoOps = jNeqSim.thermodynamicOperations.ThermodynamicOperations(fluid)
                thermoOps.calcPTphaseEnvelope(True, 0.1)

                if 0.0 in list(thermoOps.getOperation().get("dewT")):
                    thermoOps.calcPTphaseEnvelope(False, 0.1)

                dewP = np.array(thermoOps.getOperation().get("dewP"))[:-1]
                bubP = np.array(thermoOps.getOperation().get("bubP"))
                envelopeP = np.r_[dewP, bubP]

                dewT = np.array(thermoOps.getOperation().get("dewT"))[:-1]
                bubT = np.array(thermoOps.getOperation().get("bubT"))
                envelopeT = np.r_[dewT, bubT]

                probs = np.abs(np.diff(envelopeT))
                probs /= probs.sum()
                index = np.random.choice(np.arange(1, envelopeT.shape[0]), p=probs)
                T_range = np.array([envelopeT[index - 1], envelopeT[index]])
                P_range = np.array([envelopeP[index - 1], envelopeP[index]])

                T_center = (T_range[1] - T_range[0]) * np.random.random() + T_range[0]
                P_center = np.interp(T_center, T_range, P_range)

                noiseP = 0.05 * (envelopeP.max() - envelopeP.min()) * np.random.normal()
                noiseT = 0.05 * (envelopeT.max() - envelopeT.min()) * np.random.normal()
                T_sample = T_center + noiseT
                P_sample = P_center + noiseP

                fluid.setTemperature(T_sample, "K")
                fluid.setPressure(P_sample, "bara")

                TPflash(fluid)
                break
            except Exception as err:
                self.logger.error(err)
                continue

        phases = [p for p in fluid.getPhases() if p]
        phases_names = [phase.getPhaseTypeName() for phase in phases]
        return P_sample, T_sample, phases_names, fluid, composition

    def equilibrium_ratios(self, fluid: Any):
        raw_results = dataFrame(fluid)
        results = (
            raw_results.replace("", np.nan)
            .dropna(how="all")
            .iloc[1:-3, [0, 1, 2, 3]]
            .set_index(0)
            .apply(lambda serie: pd.to_numeric(serie))
        )
        results.index.name = "Component"
        results.columns = ["z", "y", "x"]

        # Phase Fractions
        V = results.at["Phase Fraction", "y"]
        L = results.at["Phase Fraction", "x"]

        # Phase Component Fractions
        phases_fractions = results.iloc[:24, :].copy()
        phases_fractions.index = [f"K_{c[1:]}" for c in NEQSIM_COMPONENTS.keys()]

        z = phases_fractions["z"]
        x = phases_fractions["x"]
        phases_fractions.loc[:, "K"] = (z - x * L) / (V * x)

        output = phases_fractions.T.loc["K", :].to_dict()
        output["nV"] = V

        return output

    def sampling(self, model_name: str):
        edl = ExperimentalDataLoader()
        datasets, _ = edl.load_cross_validation_datasets(problem="classification")
        train_dataset_size = datasets["train"][0]["features"].shape[0]

        samples = []
        num_samples = self.dataset_size * train_dataset_size
        model_folder = os.path.join("data", "models", "synthesis", "saved_models", model_name)

        st = SynthesisTraining()
        r = st.load_training_models()

        # Load generative model
        if "WGAN" in model_name:
            model_results = [m for m in r["outputs"] if m["model_name"] == model_name][0]
            latent_dim = model_results["params"]["latent_dim"]
            model = {
                "name": model_name,
                "latent_dim": latent_dim,
                "generator": tf.keras.models.load_model(os.path.join(model_folder, "final_generator.keras")),
            }
        elif "Dirichlet" in model_name:
            model = {"name": model_name, "alpha": self.load_pickle(os.path.join(model_folder, "final_alpha.pickle"))}

        # Generate gas samples #########################################################################################
        self.logger.info("Generating gas samples")
        for i in np.arange(num_samples // 3):
            self.logger.info(f"[Gas class] Using sample composition {i+1} of {num_samples // 3}")
            while True:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(model)

                if fluid.getNumberOfPhases() == 1:
                    if phases_names[0] == "gas":
                        composition_dict = {name: 100 * value for value, name in zip(composition, FEATURES_NAMES[:-2])}
                        sample_dict = {**composition_dict, "T": T_sample, "P": P_sample, "class": "gas"}
                        samples.append(sample_dict)
                        break

        # Generate oil samples #########################################################################################
        self.logger.info("Generating oil samples")
        for i in np.arange(num_samples // 3):
            self.logger.info(f"[Oil class] Using sample composition {i+1} of {num_samples // 3}")
            while True:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(model)

                if fluid.getNumberOfPhases() == 1:
                    if phases_names[0] == "oil":
                        composition_dict = {name: 100 * value for value, name in zip(composition, FEATURES_NAMES[:-2])}
                        sample_dict = {**composition_dict, "T": T_sample, "P": P_sample, "class": "oil"}
                        samples.append(sample_dict)
                        break

        # Generate mixture samples #####################################################################################
        self.logger.info("Generating mixture samples")
        for i in np.arange(num_samples // 3):
            self.logger.info(f"[Mixture class] Using sample composition {i+1} of {num_samples // 3}")
            while True:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(model)

                if fluid.getNumberOfPhases() == 2:
                    # Classification dataset
                    composition_dict = {name: 100 * value for value, name in zip(composition, FEATURES_NAMES[:-2])}
                    sample_dict = {**composition_dict, "T": T_sample, "P": P_sample, "class": "mix"}

                    # Regression dataset
                    outputs = self.equilibrium_ratios(fluid)
                    if any([v < 10e-15 for v in outputs.values()]):
                        continue
                    if any([v > 10e5 for v in outputs.values()]):
                        continue
                    else:
                        sample_dict.update(outputs)
                        samples.append(sample_dict)
                        break

        return pd.DataFrame.from_records(samples)

    def create_datasets(self, model_name: str):
        data_folder = os.path.join("data", "processed", "synthetic", model_name, f"{self.dataset_size}to1")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        for fold in range(self.k_folds):
            self.logger.info(f"Creating fold #{fold+1} dataset")

            dataset = self.sampling(model_name)
            dataset.to_csv(os.path.join(data_folder, f"train_fold={fold+1:02d}.csv"), index=False)


class DataLoader:

    def __init__(self, problem: str, model_name: str, dataset_size: int = 1):
        self.problem = problem
        synthetic_base_folder = os.path.join("data", "processed", "synthetic", model_name)
        self.synthetic_data_path = os.path.join(synthetic_base_folder, f"{dataset_size}to1")
        self.experimental_data_path = os.path.join("data", "processed", "experimental")

    def load_cross_validation_datasets(self):
        self.train_files = glob.glob(os.path.join(self.synthetic_data_path, "train_*.csv"))
        self.valid_files = glob.glob(os.path.join(self.experimental_data_path, "valid_*.csv"))
        self.test_files = glob.glob(os.path.join(self.experimental_data_path, "test_*.csv"))

        self.train_files = sorted(self.train_files, key=lambda p: int(os.path.split(p)[-1].split("=")[-1].split(".")[0]))
        self.valid_files = sorted(self.valid_files, key=lambda p: int(os.path.split(p)[-1].split("=")[-1].split(".")[0]))
        self.test_files = sorted(self.test_files, key=lambda p: int(os.path.split(p)[-1].split("=")[-1].split(".")[0]))

        datasets = {"train": [], "valid": [], "test": []}
        min_max = []
        for train_f, valid_f, test_f in zip(self.train_files, self.valid_files, self.test_files):
            train_features, train_targets = self.preprocessing(pd.read_csv(train_f), problem=self.problem)
            valid_features, valid_targets = self.preprocessing(pd.read_csv(valid_f), problem=self.problem)
            test_features, test_targets = self.preprocessing(pd.read_csv(test_f), problem=self.problem)

            if self.problem == "regression":
                min_vals, max_vals = train_targets.min(), train_targets.max()
                min_max.append([min_vals.to_numpy(), max_vals.to_numpy()])

                train_targets = (train_targets - min_vals) / (max_vals - min_vals)
                valid_targets = (valid_targets - min_vals) / (max_vals - min_vals)
                test_targets = (test_targets - min_vals) / (max_vals - min_vals)

            datasets["train"].append({"features": train_features, "targets": train_targets})
            datasets["valid"].append({"features": valid_features, "targets": valid_targets})
            datasets["test"].append({"features": test_features, "targets": test_targets})

        return datasets, min_max

    def preprocessing(self, data: pd.DataFrame, problem: str):
        processed_data = data.copy()
        processed_data[FEATURES_NAMES[:-2]] = processed_data[FEATURES_NAMES[:-2]] / 100.0

        if problem in ["classification", "regression"]:
            P_min, P_max = P_MIN_MAX
            T_min, T_max = T_MIN_MAX
            processed_data["P"] = (processed_data["P"] - P_min) / (P_max - P_min)
            processed_data["T"] = (processed_data["T"] - T_min) / (T_max - T_min)

            if problem == "regression":
                processed_data = processed_data[processed_data["class"] == "mix"]

            features = processed_data[FEATURES_NAMES].copy()

        if problem == "classification":
            targets = pd.get_dummies(processed_data["class"], dtype=np.float32)
        elif problem == "regression":
            targets = processed_data[REGRESSION_TARGET_NAMES]
        elif problem == "synthesis":
            features = processed_data[FEATURES_NAMES[:-2]].copy()
            targets = None
        return features, targets
