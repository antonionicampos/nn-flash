import matplotlib

matplotlib.use("agg")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import textwrap

from datetime import date
from src.data.handlers import DataTransform, CrossValidation
from src.models.classification.evaluate_models import ClassificationAnalysis
from src.models.classification.train_models import ClassificationTraining
from src.models.regression.evaluate_models import RegressionAnalysis
from src.models.regression.train_models import RegressionTraining
from src.models.synthesis.train_models import SynthesisTraining
from src.visualization.classification import ClassificationViz
from src.visualization.regression import RegressionViz


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

logger_params = {
    "format": "[%(asctime)s] %(levelname)s %(name)s:%(lineno)3d | %(message)s",
    "filename": logs_filepath,
}

if __name__ == "__main__":
    parser.add_argument(
        "-s",
        "--samples-per-composition",
        default="3",
        action="store",
        choices=["3", "30"],
        required=False,
        help="Select dataset depending on number of P, T samples per composition sample",
    )
    parser.add_argument(
        "--task",
        default=None,
        action="store",
        choices=["classification", "regression", "synthesis"],
        required=True,
        help="Task(s) to run pipeline",
    )
    parser.add_argument("-r", "--read", default=None, action="store_true", help="Read, transform and process raw data")
    parser.add_argument("-cv", "--cross-validation", default=None, action="store_true", help="Create CV datasets")
    parser.add_argument("-t", "--training", default=None, action="store_true", help="Do train step")
    parser.add_argument("-a", "--analysis", default=None, action="store_true", help="Do Analysis Step")
    parser.add_argument("-v", "--viz", default=None, action="store_true", help="Create and save visualizations")

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

        if args.task == "classification":
            logger.info("Starting create classification cross-validation datasets")
            cv_data.create_datasets(model="classification", samples_per_composition=samples_per_composition)
        elif args.task == "regression":
            logger.info("Starting create regression cross-validation datasets")
            cv_data.create_datasets(model="regression", samples_per_composition=samples_per_composition)
        elif args.task == "synthesis":
            logger.info("Starting create synthesis cross-validation datasets")
            cv_data.create_datasets(model="synthesis")

    if args.training:
        logger.info("Starting train models")

        if args.task == "classification":
            logger.info("Starting classification models training")
            classification_training = ClassificationTraining(samples_per_composition=samples_per_composition)
            classification_training.run()
        elif args.task == "regression":
            logger.info("Starting regression models training")
            regression_training = RegressionTraining(samples_per_composition=samples_per_composition)
            regression_training.run()
        elif args.task == "synthesis":
            logger.info("Starting synthesis models training")
            synthesis_training = SynthesisTraining()
            synthesis_training.run()

    if args.analysis:
        logger.info("Starting analyze models")

        if args.task == "classification":
            classification_analysis = ClassificationAnalysis(samples_per_composition=samples_per_composition)
            classification_analysis.run()
        elif args.task == "regression":
            regression_analysis = RegressionAnalysis(samples_per_composition=samples_per_composition)
            regression_analysis.run()

    if args.viz:
        logger.info("Starting create visualization")

        if args.task == "classification":
            classification_viz = ClassificationViz(samples_per_composition=samples_per_composition)
            classification_viz.create()
        elif args.task == "regression":
            regression_viz = RegressionViz(samples_per_composition=samples_per_composition)
            regression_viz.create()
