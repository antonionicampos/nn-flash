import glob
import json
import pandas as pd

fuild_components = [
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


def same_components(components):
    for component in components:
        if not component in fuild_components:
            return False
    return True


data = []
for filename in glob.glob("*.json"):
    with open(filename) as f:
        raw_data = json.load(f)["Fluids"]
        field = filename.split("\\")[-1].split(".")[0][11:]
        for fluid in raw_data:
            hydrocarbon_analysis = fluid["HydrocarbonAnalysis"]
            if hydrocarbon_analysis["AtmosphericFlashTestAndCompositionalAnalysis"]:
                for flash in fluid["AtmosphericFlashTestAndCompositionalAnalysis"]:
                    comp = [c["FluidComponent"] for c in flash["CompositionalAnalysis"]]
                    if same_components(comp):
                        # keys: 'UniqueID', 'LastUpdateDate', 'ReservoirFluidKind', 'LastFluidComponent', 'CompositionalAnalysis'
                        info = {
                            "Field": field,
                            "Id": flash["UniqueID"],
                            "Date": flash["LastUpdateDate"],
                            "FluidKind": flash["ReservoirFluidKind"],
                            "LastFluidMolecularWeight": flash["LastFluidComponent"]["MolecularWeight"],
                            "LastFluidSpecificGravity": flash["LastFluidComponent"]["SpecificGravity"]
                        }
                        for component in flash["CompositionalAnalysis"]:
                            info[f"z{component['FluidComponent']}"] = component["OverallComposition"]
                            if component["IsLastFluidComponent"]:
                                info["LastFluidComponent"] = component["FluidComponent"]

                        data.append(info)

pd.DataFrame.from_records(data).to_csv("raw_data.csv", index=False)
