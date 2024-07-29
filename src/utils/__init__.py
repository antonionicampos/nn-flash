import neqsim
import numpy as np
import pandas as pd

from src.utils.constants import NEQSIM_COMPONENTS, FEATURES_NAMES, P_MIN_MAX, T_MIN_MAX, REGRESSION_TARGET_NAMES
from typing import Dict


def create_fluid(composition: Dict[str, float]):
    """Create NeqSim fluid adding its components and fractions.

    Parameters
    ----------
    composition : Dict[str, float]
        fluid components with its fractions

    Returns
    -------
        NeqSim fluid.
    """
    fluid = neqsim.thermo.fluid("pr")  # Peng-Robinson EOS
    for component, fraction in composition.items():
        fluid.addComponent(NEQSIM_COMPONENTS[component], fraction)
    fluid.setMixingRule("classic")  # classic will use binary kij
    return fluid


def preprocessing(data, problem: str):
    """Preprocessing data for classification or regression problem returns features and labels.

    Parameters
    ----------
    problem : str
        classification or regression problem preprocessing data
    Returns
    -------
        features and labels
    """
    processed_data = data.copy()
    processed_data[FEATURES_NAMES[:-2]] = processed_data[FEATURES_NAMES[:-2]] / 100.0

    P_min, P_max = P_MIN_MAX
    T_min, T_max = T_MIN_MAX
    processed_data["P"] = (processed_data["P"] - P_min) / (P_max - P_min)
    processed_data["T"] = (processed_data["T"] - T_min) / (T_max - T_min)

    features = processed_data[FEATURES_NAMES].copy()

    if problem == "classification":
        labels = pd.get_dummies(processed_data["class"], dtype=np.float32)
    elif problem == "regression":
        labels = processed_data[REGRESSION_TARGET_NAMES].copy()
    return features, labels
