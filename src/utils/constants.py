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

FEATURES_NAMES = [
    "zN2",
    "zCO2",
    "zC1",
    "zC2",
    "zC3",
    "zIC4",
    "zNC4",
    "zIC5",
    "zNC5",
    "zC6",
    "zC7",
    "zC8",
    "zC9",
    "zC10",
    "zC11",
    "zC12",
    "zC13",
    "zC14",
    "zC15",
    "zC16",
    "zC17",
    "zC18",
    "zC19",
    "zC20",
    "P",
    "T",
]

VAPOR_FRACTIONS_NAMES = [
    "yN2",
    "yCO2",
    "yC1",
    "yC2",
    "yC3",
    "yIC4",
    "yNC4",
    "yIC5",
    "yNC5",
    "yC6",
    "yC7",
    "yC8",
    "yC9",
    "yC10",
    "yC11",
    "yC12",
    "yC13",
    "yC14",
    "yC15",
    "yC16",
    "yC17",
    "yC18",
    "yC19",
    "yC20",
]

LIQUID_FRACTIONS_NAMES = [
    "xN2",
    "xCO2",
    "xC1",
    "xC2",
    "xC3",
    "xIC4",
    "xNC4",
    "xIC5",
    "xNC5",
    "xC6",
    "xC7",
    "xC8",
    "xC9",
    "xC10",
    "xC11",
    "xC12",
    "xC13",
    "xC14",
    "xC15",
    "xC16",
    "xC17",
    "xC18",
    "xC19",
    "xC20",
]

REGRESSION_TARGET_NAMES = [
    "K_N2",
    "K_CO2",
    "K_C1",
    "K_C2",
    "K_C3",
    "K_IC4",
    "K_NC4",
    "K_IC5",
    "K_NC5",
    "K_C6",
    "K_C7",
    "K_C8",
    "K_C9",
    "K_C10",
    "K_C11",
    "K_C12",
    "K_C13",
    "K_C14",
    "K_C15",
    "K_C16",
    "K_C17",
    "K_C18",
    "K_C19",
    "K_C20",
    "nV",
]

TARGET_NAMES = ["Vapor", "Mistura", "Líquido"]

P_MIN_MAX = [10., 450.]

T_MIN_MAX = [150., 1125.]

K_FOLDS = 10
