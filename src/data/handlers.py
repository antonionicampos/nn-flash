import glob
import json
import logging
import neqsim
import matplotlib.pyplot as plt
import neqsim.thermo
import numpy as np
import os
import pandas as pd

from neqsim.thermo import TPflash
from neqsim.thermo.thermoTools import dataFrame, jNeqSim
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Any, Dict, List


pd.set_option("future.no_silent_downcasting", True)

FLUID_COMPONENTS = [
    "N2",
    "CO2",
    "C1",
    "C2",
    "C3",
    "IC4",
    "NC4",
    "IC5",
    "NC5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
]

NEQSIM_COMPONENTS = {
    "zN2": "nitrogen",
    "zCO2": "CO2",
    "zC1": "methane",
    "zC2": "ethane",
    "zC3": "propane",
    "zIC4": "i-butane",
    "zNC4": "n-butane",
    "zIC5": "i-pentane",
    "zNC5": "n-pentane",
    "zC6": "n-hexane",
    "zC7": "n-heptane",
    "zC8": "n-octane",
    "zC9": "n-nonane",
    "zC10": "nC10",
    "zC11": "nC11",
    "zC12": "nC12",
    "zC13": "nC13",
    "zC14": "nC14",
    "zC15": "nC15",
    "zC16": "nC16",
    "zC17": "nC17",
    "zC18": "nC18",
    "zC19": "nC19",
    "zC20": "nC20",
}


def create_fluid(composition: Dict[str, float]):
    fluid = neqsim.thermo.fluid("pr")
    for component, fraction in composition.items():
        fluid.addComponent(NEQSIM_COMPONENTS[component], fraction)
    fluid.setMixingRule("classic")  # classic will use binary kij
    return fluid


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
    create_fluid(composition: Dict[str, float])
        create NeqSim fluid adding its components and fractions.
    PT_phase_envelope_data_filter(self, savefig: bool = False)
        remove compositions that phase envelope algorithm not converge.
    """

    def __init__(self, data_folder: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_folder = data_folder

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

            field = filename.split("\\")[-1].split(".")[0][11:]
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
        processed_data = pd.read_csv(os.path.join(self.data_folder, "processed", "processed_data.csv"))
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
                    color="C0",
                )
                ax.plot(
                    list(thermoOps.getOperation().get("bubT")),
                    list(thermoOps.getOperation().get("bubP")),
                    label="bubble point",
                    alpha=0.25,
                    color="C1",
                )

        if savefig:
            ax.set_title("PT envelope")
            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("Pressure [bar]")
            f.tight_layout()

            filepath = os.path.join("docs", "images", "phase_diagrams.png")
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
    classification_sampling(samples_per_composition)
        generate pressure and temperature samples for each composition for the classification models.
    regression_sampling(samples_per_composition)
        generate pressure and temperature samples for each composition for the regression models.
    create_cross_validation_datasets(model, samples_per_composition)
        create train, validation and test datasets using cross-validation techniques.
    """

    def __init__(self, data_folder: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_folder = data_folder
        self.processed_data = pd.read_csv(os.path.join(self.data_folder, "processed", "thermo_processed_data.csv"))

        self.random_state = 13
        self.k_folds = 10
        # P_min = 10 bara | P_max = 450 bara
        # T_min = 150 K   | T_max = 1125 K
        self.P_bounds = [10.0, 450.0]
        self.T_bounds = [150.0, 1125.0]

        components = self.processed_data.columns.str.contains("z")
        self.composition_data = self.processed_data.loc[:, components]

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
        nV = results.at["Phase Fraction", "y"]
        nL = results.at["Phase Fraction", "x"]

        # Phase Component Fractions
        phases_fractions = results.iloc[:24, :].copy()
        phases_fractions.index = [f"K_{c[1:]}" for c in NEQSIM_COMPONENTS.keys()]

        z = phases_fractions["z"]
        x = phases_fractions["x"]
        phases_fractions.loc[:, "K"] = (z - x * nL) / (nV * x)

        output = phases_fractions.T.loc["K", :].to_dict()
        output["nV"] = nV

        return output

    def classification_sampling(self, samples_per_composition: int):
        samples = []

        for i in np.arange(self.processed_data.shape[0]):
            fluid = create_fluid(self.composition_data.loc[i, :].to_dict())

            # P_min = 10 bara   T_min = 150 K
            # P_max = 450 bara  T_max = 1125 K
            gas_sample, oil_sample, mix_sample = [], [], []

            while (
                len(gas_sample) < samples_per_composition // 3
                or len(oil_sample) < samples_per_composition // 3
                or len(mix_sample) < samples_per_composition // 3
            ):
                P_sample = np.random.uniform(self.P_bounds[0], self.P_bounds[1])
                T_sample = np.random.uniform(self.T_bounds[0], self.T_bounds[1])

                fluid.setTemperature(T_sample, "K")
                fluid.setPressure(P_sample, "bara")
                TPflash(fluid)

                phases = [p for p in fluid.getPhases() if p]
                phases_names = [phase.getPhaseTypeName() for phase in phases]
                if fluid.getNumberOfPhases() == 1:
                    if phases_names[0] == "oil":
                        if len(oil_sample) < samples_per_composition // 3:
                            oil_sample.append([T_sample, P_sample])
                    if phases_names[0] == "gas":
                        if len(gas_sample) < samples_per_composition // 3:
                            gas_sample.append([T_sample, P_sample])
                elif fluid.getNumberOfPhases() == 2:
                    if len(mix_sample) < samples_per_composition // 3:
                        mix_sample.append([T_sample, P_sample])

            for s in oil_sample:
                sample_dict = self.processed_data.iloc[i, :].to_dict()
                sample_dict.update({"T": s[0], "P": s[1], "class": "oil"})
                samples.append(sample_dict)

            for s in gas_sample:
                sample_dict = self.processed_data.iloc[i, :].to_dict()
                sample_dict.update({"T": s[0], "P": s[1], "class": "gas"})
                samples.append(sample_dict)

            for s in mix_sample:
                sample_dict = self.processed_data.iloc[i, :].to_dict()
                sample_dict.update({"T": s[0], "P": s[1], "class": "mix"})
                samples.append(sample_dict)

        return pd.DataFrame.from_records(samples)

    def regression_sampling(self, samples_per_composition: int):
        samples = []

        for i in np.arange(self.processed_data.shape[0]):
            fluid = create_fluid(self.composition_data.loc[i, :].to_dict())

            composition_samples = []
            while len(composition_samples) < samples_per_composition:
                P_sample = np.random.uniform(self.P_bounds[0], self.P_bounds[1])
                T_sample = np.random.uniform(self.T_bounds[0], self.T_bounds[1])

                fluid.setTemperature(T_sample, "K")
                fluid.setPressure(P_sample, "bara")
                TPflash(fluid)

                if fluid.getNumberOfPhases() == 2:
                    sample_dict = self.processed_data.iloc[i, :].to_dict()
                    outputs = self.equilibrium_ratios(fluid)

                    if any([v < 10e-15 for v in outputs.values()]):
                        continue
                    else:
                        s = {"T": T_sample, "P": P_sample}
                        s.update(outputs)
                        sample_dict.update(s)
                        composition_samples.append(sample_dict)
            samples.extend(composition_samples)

        return pd.DataFrame.from_records(samples)

    def create_datasets(self, model: str, samples_per_composition: int):
        assert model in ["classification", "regression"], "model argument can be only 'classification' or 'regression'"
        root_folder = os.path.join(self.data_folder, "processed", "experimental")

        if model == "classification":
            samples = self.classification_sampling(samples_per_composition)

            self.logger.info(f"Using Stratified K Fold with K = {self.k_folds}")
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            folder_path = os.path.join(root_folder, "classification", f"{samples_per_composition:03d}points")

            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            samples = samples.sample(frac=1, ignore_index=True)
            for i, (train_i, test_i) in enumerate(skf.split(samples, samples["class"])):
                train_size = train_i.shape[0] - test_i.shape[0]
                train_i, valid_i = train_i[:train_size], train_i[train_size:]

                self.logger.info(f">> Fold {i+1} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                self.logger.info(f"Train: {train_i.shape[0]}, valid: {valid_i.shape[0]}, test: {test_i.shape[0]}")

                train = samples.iloc[train_i, :]
                valid = samples.iloc[valid_i, :]
                test = samples.iloc[test_i, :]
                ots = train[train["class"] == "oil"].shape[0]
                gts = train[train["class"] == "gas"].shape[0]
                mts = train[train["class"] == "mix"].shape[0]
                self.logger.info(f">> Train per class: {ots} (oil), {gts} (gas), {mts} (mix)")
                ovs = valid[valid["class"] == "oil"].shape[0]
                gvs = valid[valid["class"] == "gas"].shape[0]
                mvs = valid[valid["class"] == "mix"].shape[0]
                self.logger.info(f">> Valid per class: {ovs} (oil), {gvs} (gas), {mvs} (mix)")
                ots = test[test["class"] == "oil"].shape[0]
                gts = test[test["class"] == "gas"].shape[0]
                mts = test[test["class"] == "mix"].shape[0]
                self.logger.info(f">> Test per class: {ots} (oil), {gts} (gas), {mts} (mix)")

                train_filepath = os.path.join(folder_path, f"train_fold={i+1:02d}.csv")
                valid_filepath = os.path.join(folder_path, f"valid_fold={i+1:02d}.csv")
                test_filepath = os.path.join(folder_path, f"test_fold={i+1:02d}.csv")
                samples.iloc[train_i, :].to_csv(train_filepath, index=False)
                samples.iloc[valid_i, :].to_csv(valid_filepath, index=False)
                samples.iloc[test_i, :].to_csv(test_filepath, index=False)
                self.logger.info(f"Train data saved on file {train_filepath}")
                self.logger.info(f"Valid data saved on file {valid_filepath}")
                self.logger.info(f"Test data saved on file {test_filepath}")
        elif model == "regression":
            samples = self.regression_sampling(samples_per_composition)

            self.logger.info(f"Using K Fold with K = {self.k_folds}")
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            folder_path = os.path.join(root_folder, "regression", f"{samples_per_composition:03d}points")

            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            samples = samples.sample(frac=1, ignore_index=True)
            for i, (train_i, test_i) in enumerate(kf.split(samples)):
                train_size = train_i.shape[0] - test_i.shape[0]
                train_i, valid_i = train_i[:train_size], train_i[train_size:]

                self.logger.info(f">>>> Fold {i+1} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                self.logger.info(f"Train: {train_i.shape[0]}, valid: {valid_i.shape[0]}, test: {test_i.shape[0]}")

                train = samples.iloc[train_i, :]
                valid = samples.iloc[valid_i, :]
                test = samples.iloc[test_i, :]

                train_filepath = os.path.join(folder_path, f"train_fold={i+1:02d}.csv")
                valid_filepath = os.path.join(folder_path, f"valid_fold={i+1:02d}.csv")
                test_filepath = os.path.join(folder_path, f"test_fold={i+1:02d}.csv")
                samples.iloc[train_i, :].to_csv(train_filepath, index=False)
                samples.iloc[valid_i, :].to_csv(valid_filepath, index=False)
                samples.iloc[test_i, :].to_csv(test_filepath, index=False)
                self.logger.info(f"Train data saved on file {train_filepath}")
                self.logger.info(f"Valid data saved on file {valid_filepath}")
                self.logger.info(f"Test data saved on file {test_filepath}")
