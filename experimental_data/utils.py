import neqsim
import neqsim.thermo
import numpy as np
import pandas as pd

from neqsim.thermo.thermoTools import dataFrame
from neqsim.thermo import TPflash
from typing import Dict

COMPONENTS = {
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


def set_components(composition: Dict[str, float]):
    fluid1 = neqsim.thermo.fluid("pr")
    for component, fraction in composition.items():
        fluid1.addComponent(COMPONENTS[component], fraction)
    fluid1.setMixingRule("classic")  # classic will use binary kij
    return fluid1


def equilibrium_ratios(fluid):
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
    phases_fractions.index = [f"K_{c[1:]}" for c in COMPONENTS.keys()]

    z = phases_fractions["z"]
    x = phases_fractions["x"]
    phases_fractions.loc[:, "K"] = (z - x * nL) / (nV * x)

    output = phases_fractions.T.loc["K", :].to_dict()
    output["nV"] = nV

    return output
