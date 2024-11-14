import matplotlib

matplotlib.use("agg")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import numpy as np
import tensorflow as tf
import textwrap

from datetime import datetime
from src.data.handlers import DataTransform, CrossValidation
from src.data.synthetic_gen import DataGen
from src.models.classification.evaluate_models import ClassificationAnalysis
from src.models.classification.train_models import ClassificationTraining
from src.models.regression.evaluate_models import RegressionAnalysis
from src.models.regression.train_models import RegressionTraining
from src.models.synthesis.train_models import SynthesisTraining
from src.models.synthesis.evaluate_models import SynthesisAnalysis
from src.visualization.classification import ClassificationViz
from src.visualization.regression import RegressionViz
from src.visualization.synthesis import SynthesisViz


np.random.seed(13)
tf.random.set_seed(13)

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

dt = datetime.now().strftime("%Y%m%d%H%M%S")
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
        "--task",
        default=None,
        action="store",
        choices=["classification", "regression", "synthesis"],
        required=False,
        help="Task(s) to run pipeline",
    )
    parser.add_argument(
        "--regression-loss",
        default="mse",
        action="store",
        choices=["mse", "mse_with_constraint"],
        required=False,
        help="Regression loss function",
    )
    parser.add_argument("-r", "--read", default=None, action="store_true", help="Read, transform and process raw data")
    parser.add_argument("-cv", "--cross-validation", default=None, action="store_true", help="Create CV datasets")
    parser.add_argument("-t", "--training", default=None, action="store_true", help="Do train step")
    parser.add_argument("-a", "--analysis", default=None, action="store_true", help="Do Analysis Step")
    parser.add_argument("-v", "--viz", default=None, action="store_true", help="Create and save visualizations")
    parser.add_argument("-g", "--generate", default=None, action="store_true", help="Synthesize new samples")

    args = parser.parse_args()

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

        logger.info("Start creating cross-validation datasets")
        cv_data.create_datasets()

    if args.training:
        logger.info("Starting train models")

        if args.task == "classification":
            logger.info("Starting classification models training")
            classification_training = ClassificationTraining()
            classification_training.run()
        elif args.task == "regression":
            logger.info("Starting regression models training")
            regression_training = RegressionTraining()

            if args.regression_loss == "mse":
                logger.info("Training regression models with default loss function")
                regression_training.run()
            elif args.regression_loss == "mse_with_constraint":
                logger.info("Training regression models with constrained loss function")
                regression_training.train_mse_loss_with_soft_constraint()

        elif args.task == "synthesis":
            logger.info("Starting synthesis models training")
            synthesis_training = SynthesisTraining()
            synthesis_training.run()

    if args.analysis:
        logger.info("Starting analyze models")

        if args.task == "classification":
            classification_analysis = ClassificationAnalysis()
            classification_analysis.run()
        elif args.task == "regression":
            regression_analysis = RegressionAnalysis()
            regression_analysis.run()
        elif args.task == "synthesis":
            synthesis_analysis = SynthesisAnalysis()
            synthesis_analysis.run()

    if args.viz:
        logger.info("Starting create visualization")

        if args.task == "classification":
            logger.info("Creating classification visualizations")
            classification_viz = ClassificationViz()
            classification_viz.create()
        elif args.task == "regression":
            logger.info("Creating regression visualizations")
            regression_training = RegressionTraining()
            regression_viz = RegressionViz()

            if args.regression_loss == "mse":
                logger.info("Creating regression models with default loss function visualizations")
                regression_viz.create()
            elif args.regression_loss == "mse_with_constraint":
                logger.info("Creating regression models with constrained loss function visualizations")
                regression_training.plot_mse_loss_with_soft_constraint()

        elif args.task == "synthesis":
            logger.info("Creating synthesis visualizations")
            synthesis_viz = SynthesisViz()
            synthesis_viz.create()

    if args.generate:
        model_name = "Dirichlet Estimator"
        dataset_size = 1

        logger.info(f"Start generating synthetic samples using {model_name} and {dataset_size}x original dataset size")
        dg = DataGen(dataset_size=dataset_size)
        dg.create_datasets(problem=args.task, model_name=model_name)
