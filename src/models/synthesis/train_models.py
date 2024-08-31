import logging
import numpy as np
import os

from datetime import datetime
from dirichlet import mle
from scipy.stats import dirichlet
from src.data.handlers import DataLoader
from src.models.synthesis.experiments import hparams
from src.utils import load_model_hparams
from tqdm import tqdm


class SynthesisTraining:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_folder = os.path.join("data", "models", "synthesis", "saved_models")

    def run(self):
        data_loader = DataLoader()
        cv_data, _ = data_loader.load_cross_validation_datasets(problem="synthesis")
        train_data, valid_data = cv_data["train"], cv_data["valid"]

        results = {"outputs": []}
        for hp in load_model_hparams(hparams):
            model_name = hp["model_name"]
            model_type = hp["model_type"]

            model_results = {**hp}
            folds = []

            print(f"\nModel: {model_name}")

            self.logger.info(f"Model: {model_name}")

            training_model_start = datetime.now()
            pbar = tqdm(total=len(train_data))
            for fold, (train, valid) in enumerate(zip(train_data, valid_data)):
                pbar.set_description(f"Train using fold {fold+1} dataset")

                train_features, valid_features = train["features"], valid["features"]

                if model_type == "dirichlet":
                    alpha = mle(train_features)
                    folds.append({"fold": fold + 1, "alpha": alpha})
                    self.logger.info(f"Fold {fold+1} dataset, first 3 alphas: {alpha[:3]}")
                elif model_type == "wgan":
                    pass
                pbar.update()

            pbar.close()
            os.system("cls" if os.name == "nt" else "clear")
            model_results["folds"] = folds
            results["outputs"].append(model_results)
            self.logger.info(f"{model_name} training elapsed Time: {datetime.now() - training_model_start}")
