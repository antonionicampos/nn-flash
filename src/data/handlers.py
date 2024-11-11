import glob
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from neqsim.thermo import TPflash
from neqsim.thermo.thermoTools import dataFrame, jNeqSim
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.utils.constants import (
    FLUID_COMPONENTS,
    NEQSIM_COMPONENTS,
    FEATURES_NAMES,
    P_MIN_MAX,
    T_MIN_MAX,
    REGRESSION_TARGET_NAMES,
)
from src.utils import create_fluid
from typing import Any, List


pd.set_option("future.no_silent_downcasting", True)


class DataTransform:
    """Transform data from raw JSON files to a CSV file with processed data.

    The sequence of transformations are listed below.

    - load_raw_data() method
        1. Convert JSON data to CSV file
        2. Save raw_data.csv file

    - transform_raw_data() method
        1. Normalize compositions to sum 100%
        2. Remove samples with NaN values on composition
        3. Remove samples with last component different from Eicosane (C20)
        4. Remove samples with components with fractions equal to zero
        5. Save processed_data.csv file

    - PT_phase_envelope_data_filter() method
        1. Remove samples that NeqSim phase envelope algorithm not converged
        2. Save thermo_processed_data.csv file

    Attributes
    ----------
    logger : logging.Logger
        logger object.
    data_folder : str
        data folder path.

    Methods
    -------
    is_same_components(components: List[str])
        check if list of components has all predefined components.
    load_raw_data()
        convert raw JSON file to raw CSV file.
    transform_raw_data()
        sequence of data processing (listed above).
    PT_phase_envelope_data_filter(self, savefig: bool = False)
        remove compositions that phase envelope algorithm not converge.
    """

    def __init__(self, data_folder: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_folder = data_folder

        self.raw_folder = os.path.join("data", "raw")
        self.processed_folder = os.path.join("data", "processed")
        self.images_folder = os.path.join("data", "images")

        if not os.path.isdir(self.raw_folder):
            os.mkdir(self.raw_folder)
        if not os.path.isdir(self.processed_folder):
            os.mkdir(self.processed_folder)
        if not os.path.isdir(self.images_folder):
            os.mkdir(self.images_folder)

    def is_same_components(self, components: List[str]):
        for component in components:
            if not component in FLUID_COMPONENTS:
                return False
        return True

    def load_raw_data(self):
        data = []
        for filename in glob.glob(os.path.join(self.data_folder, "external", "*.json")):
            with open(filename) as f:
                raw_data = json.load(f)["Fluids"]

            field = os.path.split(filename)[-1].split(".")[0][11:]
            for fluid in raw_data:
                hydrocarbon_analysis = fluid["HydrocarbonAnalysis"]
                if hydrocarbon_analysis["AtmosphericFlashTestAndCompositionalAnalysis"]:
                    for flash in fluid["AtmosphericFlashTestAndCompositionalAnalysis"]:
                        last_fluid_mw = flash["LastFluidComponent"]["MolecularWeight"]
                        last_fluid_sg = flash["LastFluidComponent"]["SpecificGravity"]
                        comp = [c["FluidComponent"] for c in flash["CompositionalAnalysis"]]
                        if self.is_same_components(comp):
                            # keys: 'UniqueID', 'LastUpdateDate', 'ReservoirFluidKind', 'LastFluidComponent', 'CompositionalAnalysis'
                            info = {
                                "Field": field,
                                "Id": flash["UniqueID"],
                                "Date": flash["LastUpdateDate"],
                                "FluidKind": flash["ReservoirFluidKind"],
                                "LastFluidMolecularWeight": last_fluid_mw,
                                "LastFluidSpecificGravity": last_fluid_sg,
                            }
                            for component in flash["CompositionalAnalysis"]:
                                composition = component["OverallComposition"]
                                info[f"z{component['FluidComponent']}"] = composition
                                if component["IsLastFluidComponent"]:
                                    info["LastFluidComponent"] = component["FluidComponent"]

                            data.append(info)

        filepath = os.path.join(self.data_folder, "raw", "raw_data.csv")
        self.logger.info("Saving raw data")
        pd.DataFrame.from_records(data).to_csv(filepath, index=False)
        self.logger.info(f"Raw CSV file saved on {filepath}")

    def transform_raw_data(self):
        df = pd.read_csv(os.path.join(self.data_folder, "raw", "raw_data.csv"))
        compositions = df.iloc[:, df.columns.str.contains("z")]

        # Composition sum to 100%
        normalize = lambda row: 100 * row / compositions.sum(axis=1)
        components = df.columns.str.contains("z")
        df.iloc[:, components] = compositions.apply(normalize).round(2)

        self.logger.info(f"Initial samples: {df.shape[0]}")

        df = df.dropna()
        self.logger.info(f"Remove samples w/ NaN values. Current samples: {df.shape[0]}")

        df = df[df["LastFluidComponent"] == "C20"]
        self.logger.info(f"Remove samples w/ LastFluidComponent different from C20. Current samples: {df.shape[0]}")

        components = df.columns.str.contains("z")
        compositions = df.iloc[:, components]
        df = df[~(compositions == 0).any(axis=1)]
        self.logger.info(f"Remove samples w/ values equal to zero. Current samples: {df.shape[0]}")

        filepath = os.path.join(self.data_folder, "processed", "processed_data.csv")
        self.logger.info("Saving transformed data")
        df.to_csv(filepath, index=False)
        self.logger.info(f"Transformed CSV file saved on {filepath}")

    def PT_phase_envelope_data_filter(self, savefig: bool = False):
        processed_data = pd.read_csv(os.path.join(self.processed_folder, "processed_data.csv"))
        composition_data = processed_data.loc[:, processed_data.columns.str.contains("z")]

        self.logger.info(f"Initial samples: {processed_data.shape[0]}")

        f, ax = plt.subplots()
        for i in np.arange(processed_data.shape[0]):
            fluid = create_fluid(composition_data.loc[i, :].to_dict())
            thermoOps = jNeqSim.thermodynamicOperations.ThermodynamicOperations(fluid)
            thermoOps.calcPTphaseEnvelope(True, 0.1)

            if 0.0 in list(thermoOps.getOperation().get("dewT")):
                thermoOps.calcPTphaseEnvelope(False, 0.1)
                if 0.0 in list(thermoOps.getOperation().get("dewT")):
                    composition_data = composition_data.drop(i)
                    processed_data = processed_data.drop(i)
                    continue

            if savefig:
                ax.plot(
                    list(thermoOps.getOperation().get("dewT")),
                    list(thermoOps.getOperation().get("dewP")),
                    label="dew point",
                    alpha=0.25,
                    color="darkblue",
                )
                ax.plot(
                    list(thermoOps.getOperation().get("bubT")),
                    list(thermoOps.getOperation().get("bubP")),
                    label="bubble point",
                    alpha=0.25,
                    color="green",
                )

        if savefig:
            ax.set_title("Diagrama de fases PT")
            ax.set_xlabel("Temperatura [K]")
            ax.set_ylabel("Pressão [bara]")
            f.tight_layout()

            filepath = os.path.join(self.images_folder, "phase_diagrams.png")
            f.savefig(filepath, dpi=600)

        processed_data = processed_data.reset_index(drop=True)
        self.logger.info(f"Remove not converged samples. Current samples: {processed_data.shape[0]}")
        filepath = os.path.join(self.data_folder, "processed", "thermo_processed_data.csv")
        self.logger.info("Saving converged phase diagrams data")
        processed_data.to_csv(filepath, index=False)
        self.logger.info(f"Transformed CSV file saved on {filepath}")


class CrossValidation:
    """Generate cross-validation datasets for the classification and regression problems.

    Sample pressures and temperatures for classification and regression models and create cross-validation datasets
    using Stratified K Fold for classification taking the classes (mix, gas and oil) as stratification weights and
    using K Fold for regression.

    Attributes
    ----------
    logger : logging.Logger
        logger object.
    data_folder : str
        data folder path.
    processed_data : pd.DataFrame
        processed data.
    random_state : int
        random state used on scikit-learn random operations.
    k_folds : int
        number of folds used on cross-validation dataset split.
    P_bounds : List[float]
        minimum and maximum pressure boundaries in bara.
    T_bounds : List[float]
        minimum and maximum temperature boundaries in K.
    composition_data : pd.DataFrame
        composition data extracted from processed data.

    Methods
    -------
    equilibrium_ratios(fluid)
        return equilibrium ratios of every component.
    generate_sample(composition)
        generate PT and fluid using neqsim simulator.
    sampling()
        generate pressure and temperature samples for each composition for the classification and regression models.
    create_cross_validation_datasets(model)
        create train, validation and test datasets using cross-validation techniques.
    """

    def __init__(self, data_folder: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_folder = data_folder
        self.processed_data = pd.read_csv(os.path.join(self.data_folder, "processed", "thermo_processed_data.csv"))
        self.processed_data = self.processed_data.drop_duplicates(subset=FEATURES_NAMES[:-2], ignore_index=True)

        self.random_state = 13
        self.k_folds = 10
        # P_min = 10 bara | P_max = 450 bara
        # T_min = 150 K   | T_max = 1125 K
        self.P_bounds = [10.0, 450.0]
        self.T_bounds = [150.0, 1125.0]

        components = self.processed_data.columns.str.contains("z")
        self.composition_data = self.processed_data.loc[:, components]

        self.folder_path = os.path.join(self.data_folder, "processed", "experimental")
        if not os.path.isdir(self.folder_path):
            os.makedirs(self.folder_path)

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
    
    def generate_sample(self, composition):
        P_sample = np.random.uniform(self.P_bounds[0], self.P_bounds[1])
        T_sample = np.random.uniform(self.T_bounds[0], self.T_bounds[1])

        composition_dict = {name: value for value, name in zip(composition, FEATURES_NAMES[:-2])}

        fluid = create_fluid(composition_dict)
        fluid.setTemperature(T_sample, "K")
        fluid.setPressure(P_sample, "bara")
        TPflash(fluid)

        phases = [p for p in fluid.getPhases() if p]
        phases_names = [phase.getPhaseTypeName() for phase in phases]
        return P_sample, T_sample, phases_names, fluid, composition

    def sampling(self):
        num_samples_per_composition = 10
        samples = []

        self.logger.info("Start sampling...")
        for i in np.arange(self.processed_data.shape[0]):
            self.logger.info(f"Using sample composition {i+1} of {self.processed_data.shape[0]}")
            gas_sample, oil_sample, mix_sample = 0, 0, 0

            composition = self.processed_data.loc[i, FEATURES_NAMES[:-2]]

            # Generate gas samples #####################################################################################
            while gas_sample < num_samples_per_composition:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(composition)

                if fluid.getNumberOfPhases() == 1:
                    if phases_names[0] == "gas":
                        sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "gas"}
                        samples.append(sample_dict)
                        gas_sample += 1

            # Generate oil samples #####################################################################################
            while oil_sample < num_samples_per_composition:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(composition)

                if fluid.getNumberOfPhases() == 1:
                    if phases_names[0] == "oil":
                        sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "oil"}
                        samples.append(sample_dict)
                        oil_sample += 1

            # Generate mixture samples #################################################################################
            while mix_sample < num_samples_per_composition:
                P_sample, T_sample, phases_names, fluid, composition = self.generate_sample(composition)

                if fluid.getNumberOfPhases() == 2:
                    # Classification dataset
                    sample_dict = {**composition, "T": T_sample, "P": P_sample, "class": "mix"}

                    # Regression dataset
                    outputs = self.equilibrium_ratios(fluid)
                    if any([v < 10e-15 for v in outputs.values()]):
                        continue
                    else:
                        sample_dict.update(outputs)
                        samples.append(sample_dict)
                        mix_sample += 1

        samples = pd.DataFrame.from_records(samples)
        
        # Save/return dataset
        samples.to_csv(os.path.join(self.folder_path, "dataset.csv"), index=False)
        return samples

    def create_datasets(self):
        if os.path.isfile(os.path.join(self.folder_path, "dataset.csv")):
            samples = pd.read_csv(os.path.join(self.folder_path, "dataset.csv"))
        else:
            samples = self.sampling()

        self.logger.info(f"Using Stratified K Fold with K = {self.k_folds}")
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        samples = samples.sample(frac=1, ignore_index=True)

        for i, (train_idx, test_idx) in enumerate(skf.split(samples, samples["class"])):
            train_data = samples.iloc[train_idx, :]
            test_data = samples.iloc[test_idx, :]

            valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=13, stratify=test_data["class"])

            train_data.to_csv(os.path.join(self.folder_path, f"train_fold={i+1:02d}.csv"), index=False)
            valid_data.to_csv(os.path.join(self.folder_path, f"valid_fold={i+1:02d}.csv"), index=False)
            test_data.to_csv(os.path.join(self.folder_path, f"test_fold={i+1:02d}.csv"), index=False)
    

class DataLoader:
    def __init__(self):
        self.processed_path = os.path.join("data", "processed")
        self.raw_path = os.path.join("data", "raw")

    def load_cross_validation_datasets(self, problem: str, samples_per_composition: int = None):
        problem_type = ["classification", "regression", "synthesis"]
        assert problem in problem_type, "problem parameter can only be 'classification', 'regression' or 'synthesis'"

        if problem in ["classification", "regression"] and samples_per_composition:
            cv_folder = os.path.join(
                self.processed_path,
                "experimental",
                problem,
                f"{samples_per_composition:03d}points",
            )
        else:
            cv_folder = os.path.join(self.processed_path, "experimental", problem)

        self.train_files = glob.glob(os.path.join(cv_folder, "train_*.csv"))
        self.valid_files = glob.glob(os.path.join(cv_folder, "valid_*.csv"))
        self.test_files = glob.glob(os.path.join(cv_folder, "test_*.csv"))

        datasets = {"train": [], "valid": [], "test": []}
        min_max = []
        for train_f, valid_f, test_f in zip(self.train_files, self.valid_files, self.test_files):

            train_features, train_targets = self.preprocessing(pd.read_csv(train_f), problem=problem)
            valid_features, valid_targets = self.preprocessing(pd.read_csv(valid_f), problem=problem)
            test_features, test_targets = self.preprocessing(pd.read_csv(test_f), problem=problem)

            if problem == "regression":
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
        processed_data = processed_data.drop_duplicates(subset=FEATURES_NAMES[:-2], ignore_index=True)
        processed_data[FEATURES_NAMES[:-2]] = processed_data[FEATURES_NAMES[:-2]] / 100.0

        if problem in ["classification", "regression"]:
            P_min, P_max = P_MIN_MAX
            T_min, T_max = T_MIN_MAX
            processed_data["P"] = (processed_data["P"] - P_min) / (P_max - P_min)
            processed_data["T"] = (processed_data["T"] - T_min) / (T_max - T_min)
            
            features = processed_data[FEATURES_NAMES].copy()

        if problem == "classification":
            targets = pd.get_dummies(processed_data["class"], dtype=np.float32)
        elif problem == "regression":
            targets = processed_data[REGRESSION_TARGET_NAMES]
        elif problem == "synthesis":
            features = processed_data[FEATURES_NAMES[:-2]].copy()
            targets = None
        return features, targets

    def load_processed_dataset(self):
        return pd.read_csv(os.path.join(self.processed_path, "thermo_processed_data.csv"))

    def load_raw_dataset(self):
        return pd.read_csv(os.path.join(self.raw_path, "raw_data.csv"))
