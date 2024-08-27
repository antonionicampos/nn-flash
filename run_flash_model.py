import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from src.data.handlers import DataLoader
from src.models import FlashModel
from src.utils.constants import VAPOR_FRACTIONS_NAMES, LIQUID_FRACTIONS_NAMES

samples_per_composition = 30
fold = 0
c_model = "Rede Neural #16"
r_model = "Rede Neural #16"

# Dataset
dl = DataLoader()
datasets, min_max = dl.load_cross_validation_datasets("regression", samples_per_composition)
features: pd.DataFrame = pd.concat([ds["features"] for ds in datasets["valid"]], ignore_index=True)
print(f"Features shape: {features.shape}")

min_max = np.array(min_max)
min_vals = min_max[:, 0, :].min(axis=0)
max_vals = min_max[:, 1, :].max(axis=0)

# Model
models_folder = os.path.join("data", "models")
c_folder = os.path.join(models_folder, "classification", "saved_models", "030points", c_model, f"Fold{fold+1}")
r_folder = os.path.join(models_folder, "regression", "saved_models", "030points", r_model, f"Fold{fold+1}")

stability_model = tf.keras.models.load_model(os.path.join(c_folder, "model.keras"))
phase_model = tf.keras.models.load_model(os.path.join(r_folder, "model.keras"))

flash_model = FlashModel(stability_model, phase_model, min_vals, max_vals)
start = datetime.now()
outputs = flash_model(features)
print(f"elapsed time: {datetime.now() - start}")

print(end="\n\n")
print(outputs)
