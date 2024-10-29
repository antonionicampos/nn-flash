import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from IPython.display import clear_output
from neqsim.thermo import TPflash
from neqsim.thermo.thermoTools import dataFrame
from src.models.synthesis.train_models import SynthesisTraining
from src.utils import create_fluid
from src.utils.constants import NEQSIM_COMPONENTS, FEATURES_NAMES
from typing import Any


class DataGen:

    def __init__(self, dataset_size: int = 1):
        self.logger = logging.getLogger(__name__)
        self.random_state = 13
        self.k_folds = 10
        self.dataset_size = dataset_size

        # P_min = 10 bara | P_max = 450 bara
        # T_min = 150 K   | T_max = 1125 K
        self.P_bounds = [10.0, 450.0]
        self.T_bounds = [150.0, 1125.0]

    def classification_sampling(self, model_name: str = "WGAN #9"):
        samples = []
        num_samples = self.dataset_size * 10656

        st = SynthesisTraining()
        results = st.load_training_models()
        model_results = [model for model in results["outputs"] if model["model_name"] == model_name][0]
        latent_dim = model_results["params"]["latent_dim"]
        model_folder = os.path.join("data", "models", "synthesis", "saved_models", model_name)
        generator = tf.keras.models.load_model(os.path.join(model_folder, "final_generator.keras"))

        # P_min = 10 bara   T_min = 150 K
        # P_max = 450 bara  T_max = 1125 K
        gas_sample, oil_sample, mix_sample = 0, 0, 0

        while (gas_sample < num_samples // 3) or (oil_sample < num_samples // 3) or (mix_sample < num_samples // 3):
            P_sample = np.random.uniform(self.P_bounds[0], self.P_bounds[1])
            T_sample = np.random.uniform(self.T_bounds[0], self.T_bounds[1])

            x = tf.random.normal([1, latent_dim])
            composition_array = generator(x).numpy().flatten()
            composition = {name: 100 * value for value, name in zip(composition_array, FEATURES_NAMES[:-2])}

            fluid = create_fluid(composition)

            fluid.setTemperature(T_sample, "K")
            fluid.setPressure(P_sample, "bara")
            TPflash(fluid)

            phases = [p for p in fluid.getPhases() if p]
            phases_names = [phase.getPhaseTypeName() for phase in phases]

            if fluid.getNumberOfPhases() == 1:
                if phases_names[0] == "oil":
                    if oil_sample < num_samples // 3:
                        sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "oil"}
                        samples.append(sample_dict)
                        oil_sample += 1
                if phases_names[0] == "gas":
                    if gas_sample < num_samples // 3:
                        sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "gas"}
                        samples.append(sample_dict)
                        gas_sample += 1
            elif fluid.getNumberOfPhases() == 2:
                if mix_sample < num_samples // 3:
                    sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "mix"}
                    samples.append(sample_dict)
                    mix_sample += 1

            clear_output(wait=True)
            print(f"Gas: {gas_sample}, Oil: {oil_sample}, Mix: {mix_sample}")

        return pd.DataFrame.from_records(samples)

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

    def regression_sampling(self, model_name: str = "WGAN #9"):
        samples = []
        num_samples = self.dataset_size * 10656

        st = SynthesisTraining()
        results = st.load_training_models()
        model_results = [model for model in results["outputs"] if model["model_name"] == model_name][0]
        latent_dim = model_results["params"]["latent_dim"]
        model_folder = os.path.join("data", "models", "synthesis", "saved_models", model_name)
        generator = tf.keras.models.load_model(os.path.join(model_folder, "final_generator.keras"))

        composition_samples = 0

        while composition_samples < num_samples:
            P_sample = np.random.uniform(self.P_bounds[0], self.P_bounds[1])
            T_sample = np.random.uniform(self.T_bounds[0], self.T_bounds[1])

            x = tf.random.normal([1, latent_dim])
            composition_array = generator(x).numpy().flatten()
            composition = {name: 100 * value for value, name in zip(composition_array, FEATURES_NAMES[:-2])}

            fluid = create_fluid(composition)

            fluid.setTemperature(T_sample, "K")
            fluid.setPressure(P_sample, "bara")
            TPflash(fluid)

            if fluid.getNumberOfPhases() == 2:
                outputs = self.equilibrium_ratios(fluid)

                if any([v < 10e-15 for v in outputs.values()]):
                    continue
                else:
                    composition.update({"T": T_sample, "P": P_sample, **outputs})
                    samples.append(composition)
                    composition_samples += 1
                    clear_output()
                    print(f"Samples created: {composition_samples}")

        return pd.DataFrame.from_records(samples)

    def create_datasets(self, model: str):
        base_folder = os.path.join("data", "processed", "synthetic")
        data_path = os.path.join(base_folder, model, f"{self.dataset_size}to1")

        for fold in range(self.k_folds):
            if model == "classification":
                dataset = self.classification_sampling()
                dataset.to_csv(os.path.join(data_path, f"train_fold={fold+1:02d}.csv"))
            elif model == "regression":
                pass
