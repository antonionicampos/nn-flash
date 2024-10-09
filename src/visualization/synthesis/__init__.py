import logging
import os
import pandas as pd

from src.data.handlers import DataLoader
from src.models.synthesis.train_models import SynthesisTraining


class SynthesisViz:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.k_folds = 5
        training = SynthesisTraining()
        self.data_loader = DataLoader()

        cv_data, _ = self.data_loader.load_cross_validation_datasets(problem="synthesis")
        self.valid_data = cv_data["valid"]
        self.results = training.load_training_models()
        self.viz_folder = os.path.join("data", "visualization", "synthesis", "saved_viz")

        if not os.path.isdir(self.viz_folder):
            os.makedirs(self.viz_folder)

    def models_table(self):
        outputs = self.results["outputs"]
        outputs = [
            {"model_id": model["model_id"], "model_name": model["model_name"], **model["params"], **model["opt"]}
            for model in outputs
        ]

        table = pd.DataFrame.from_records(outputs)
        table.to_latex(os.path.join(self.viz_folder, "models_table.tex"), index=False)

    def performance_indices_table(self):
        pass

    def errorbar_plot(self):
        pass

    def create(self):
        pass
