import logging
import numpy as np
import os
import tensorflow as tf

from datetime import datetime
from src.data.handlers import DataLoader
from src.models.classification import NeuralNetClassifier
from src.models.classification.utils import preprocessing, save_training_models
from src.models.classification.inputs import hparams
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)
np.random.seed(13)
tf.random.set_seed(13)


def run_training():
    logger = logging.getLogger(__name__)

    data_loader = DataLoader(problem="classification")
    cv_data = data_loader.load_cross_validation_datasets()

    train_data, valid_data = cv_data["train"], cv_data["valid"]

    results = {}
    training_start = datetime.now()
    for hp in hparams:
        model_name = hp["model_name"]
        model_id = hp["id"]
        arch_params = hp["arch"]
        opt_params = hp["opt"]

        learning_rate = opt_params["lr"]
        epochs = opt_params["epochs"]
        batch_size = opt_params["batch_size"]

        train_results = {"model_id": model_id, "arch": arch_params, "opt": opt_params}
        folds = []

        print(f"\nModel: {model_name}")
        print(f"    Archtecture Params: {arch_params}")
        print(f"    Optimization Params: {opt_params}", end="\n\n")

        logger.info(f"Model: {model_name}")
        logger.info(f"Archtecture Params: {arch_params}")
        logger.info(f"Optimization Params: {opt_params}")

        training_model_start = datetime.now()
        pbar = tqdm(total=len(train_data))
        for fold, (train, valid) in enumerate(zip(train_data, valid_data)):
            pbar.set_description(f"Fold {fold+1} dataset")
            logger.info(f"Fold {fold+1} dataset")

            train_features, train_labels = preprocessing(train)
            valid_features, valid_labels = preprocessing(valid)

            features, labels = train_features.values, train_labels.values
            train_ds = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(10000).batch(batch_size)

            features, labels = valid_features.values, valid_labels.values
            valid_ds = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            accuracy = tf.keras.metrics.CategoricalAccuracy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(),
                tf.keras.callbacks.EarlyStopping(min_delta=0.005, patience=10),
            ]
            model = NeuralNetClassifier(**arch_params)
            model.compile(optimizer=optimizer, loss=loss_object, metrics=[accuracy])
            history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=valid_ds,
                callbacks=callbacks,
                verbose=0,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
            )
            folds.append({"fold": fold + 1, "model": model, "history": history.history})
            logger.info({k: np.round(v[-1], decimals=4) for k, v in history.history.items()})
            pbar.update()

        pbar.close()
        os.system("cls")
        train_results["folds"] = folds
        results[model_name] = train_results
        logger.info(f"{model_name} training elapsed Time: {datetime.now() - training_model_start}")

    logger.info("Saving models")
    logger.info(f"Total elapsed Time: {datetime.now() - training_start}")
    save_training_models(results)
