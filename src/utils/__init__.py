import neqsim
import numpy as np

from src.utils.constants import NEQSIM_COMPONENTS
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


def denorm(data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray):
    return data * (max_vals - min_vals) + min_vals
