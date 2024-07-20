import argparse
import logging
import os
import textwrap

from src.data.handlers import DataTransform, CrossValidation


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

logs_filepath = os.path.join(logs_folder, "main.log")
if not os.path.isdir(logs_folder):
    os.mkdir(logs_folder)

neqsim_logger = logging.getLogger("main")
neqsim_logger.propagate = False

logger_params = {"format": "[%(asctime)s] %(levelname)s %(name)s:%(lineno)3d | %(message)s", "filename": logs_filepath}

if __name__ == "__main__":
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
    parser.add_argument("-t", "--training", default=None, action="store_true", help="Do train step")
    parser.add_argument("-w", "--warning", default=None, action="store_true", help="Run main.py in WARNING mode")
    parser.add_argument("-v", "--verbose", default=None, action="store_true", help="Run main.py in DEBUG mode")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, **logger_params)
        logging.info("Running in DEBUG mode")
    elif args.warning:
        logging.basicConfig(level=logging.WARNING, **logger_params)
        logging.info("Running in WARNING mode")
    else:
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
        samples_per_composition = 3
        
        logger.info("Starting create cross-validation data")
        cv_data = CrossValidation(data_folder)

        logger.info("Starting create classification cross-validation datasets")
        cv_data.create_datasets(model="classification", samples_per_composition=samples_per_composition)
        logger.info("Starting create regression cross-validation datasets")
        cv_data.create_datasets(model="regression", samples_per_composition=samples_per_composition)

    if args.training:
        logger.info("Starting training models")
