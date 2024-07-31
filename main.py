import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import textwrap

from datetime import date
from src.data.handlers import DataTransform, CrossValidation
from src.models.classification.evaluate_models import ClassificationAnalysis
from src.models.classification.train_models import ClassificationTraining
from src.models.regression.train_models import RegressionTraining
from src.visualization.classification import Viz


parser = argparse.ArgumentParser(
    prog="NN Flash",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(
        """\
        Neural Networks for Flash Calculations.
        https://github.com/antonionicampos/nn-flash
    """
    ),
)

logs_folder = os.path.join("data", "logs")
data_folder = os.path.join("data")

dt = date.today().strftime("%Y%m%d")
logs_filepath = os.path.join(logs_folder, f"main_{dt}.log")
if not os.path.isdir(logs_folder):
    os.mkdir(logs_folder)

neqsim_logger = logging.getLogger("main")
tensorflow_logger = logging.getLogger("tensorflow")
matplotlib_textmanager_logger = logging.getLogger("matplotlib.texmanager")
neqsim_logger.propagate = False
tensorflow_logger.propagate = False
matplotlib_textmanager_logger.propagate = False

logger_params = {"format": "[%(asctime)s] %(levelname)s %(name)s:%(lineno)3d | %(message)s", "filename": logs_filepath}

if __name__ == "__main__":
    parser.add_argument(
        "-s",
        "--samples-per-composition",
        default=3,
        action="store",
        choices=["3", "30", "300"],
        help="Select dataset depending on number of P, T samples per composition sample",
    )
    parser.add_argument(
        "-r",
        "--read",
        default=None,
        action="store_true",
        help="Read, transform and process raw data",
    )
    parser.add_argument(
        "-cv",
        "--cross-validation",
        default=None,
        action="store_true",
        help="Create CV datasets from processed data",
    )
    parser.add_argument(
        "-t",
        "--training",
        default=None,
        action="store_true",
        help="Do train step",
    )
    parser.add_argument(
        "-a",
        "--analysis",
        default=None,
        action="store_true",
        help="Do Analysis Step",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        default=None,
        action="store_true",
        help="Create and save visualizations",
    )

    args = parser.parse_args()
    samples_per_composition = int(args.samples_per_composition)

    logging.basicConfig(level=logging.INFO, **logger_params)
    logging.info("Running in INFO mode")

    logger = logging.getLogger(__name__)

    if args.read:
        logger.info("Starting read data")
        data_transform = DataTransform(data_folder)
        logger.info("Starting read raw data from JSON file and transform to CSV file")
        data_transform.load_raw_data()
        logger.info("Starting transform CSV raw data")
        data_transform.transform_raw_data()
        logger.info("Start filter not converged PT Phase Diagrams samples")
        data_transform.PT_phase_envelope_data_filter(savefig=True)

    if args.cross_validation:
        logger.info("Starting create cross-validation data")
        cv_data = CrossValidation(data_folder)
        logger.info("Starting create classification cross-validation datasets")
        cv_data.create_datasets(model="classification", samples_per_composition=samples_per_composition)
        logger.info("Starting create regression cross-validation datasets")
        cv_data.create_datasets(model="regression", samples_per_composition=samples_per_composition)

    if args.training:
        logger.info("Starting train models")
        classification_training = ClassificationTraining(samples_per_composition=samples_per_composition)
        regression_training = RegressionTraining(samples_per_composition=samples_per_composition)
        # classification_training.run()
        regression_training.run()

    if args.analysis:
        logger.info("Starting analyze models")
        analysis = ClassificationAnalysis(samples_per_composition=samples_per_composition)
        analysis.run()

    if args.visualization:
        logger.info("Starting create visualization")
        viz = Viz(samples_per_composition=samples_per_composition)
        viz.create()
        logger.info("Starting generate Phase Diagram for classification models")

        model_ids = [1, 13]
        label = 1  # Mix

        viz = Viz(samples_per_composition=samples_per_composition)
        viz.phase_diagram(
            model_ids=model_ids,
            samples_per_composition=samples_per_composition,
            label=label,
            use_mean_prediction=True,
        )
