import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from classification.mlp.models import MLPClassifier
from tqdm.notebook import tqdm
from typing import List, Any, Dict

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


def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def preprocessing(data):
    data[FEATURES_NAMES[:-2]] = data[FEATURES_NAMES[:-2]] / 100.0
    # data[["P", "T"]] = normalize(data[["P", "T"]])

    # P_sample (min, max): (10, 450)
    # T_sample (min, max): (150, 1125)
    data["P"] = (data["P"] - 10.0) / (450.0 - 10.0)
    data["T"] = (data["T"] - 150.0) / (1125.0 - 150.0)

    features = data[FEATURES_NAMES].copy()
    labels = pd.get_dummies(data["class"], dtype=np.float32)
    return features, labels


def training_model(train_files, valid_files, **kwargs):
    model_id = kwargs["id"]
    model_name = kwargs["model_name"]
    arch_params = kwargs["arch"]
    opt_params = kwargs["opt"]

    learning_rate = opt_params["lr"]
    epochs = opt_params["epochs"]
    batch_size = opt_params["batch_size"]

    pbar = tqdm(total=len(train_files))
    results = []

    print(f"Model: {model_name}")
    print(f"    Archtecture Params: {arch_params}")
    print(f"    Optimization Params: {opt_params}", end="\n\n")
    for train_f, valid_f in zip(train_files, valid_files):
        description = train_f.split("\\")[-1].split(".")[0]
        pbar.set_description(description)

        train_data = pd.read_csv(train_f)
        valid_data = pd.read_csv(valid_f)

        train_features, train_labels = preprocessing(train_data)
        valid_features, valid_labels = preprocessing(valid_data)

        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (train_features.values, train_labels.values)
            )
            .shuffle(10000)
            .batch(batch_size)
        )

        valid_ds = tf.data.Dataset.from_tensor_slices(
            (valid_features.values, valid_labels.values)
        ).batch(batch_size)

        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.EarlyStopping(min_delta=0.005, patience=10),
        ]
        model = MLPClassifier(**arch_params)
        model.compile(optimizer=optimizer, loss=loss_object, metrics=[accuracy])
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=callbacks,
            verbose=1,
        )
        results.append(
            {
                "id": model_id,
                "model_name": model_name,
                "arch": arch_params,
                "opt": opt_params,
                "data": description.split("\\")[-1],
                "model": model,
                "history": history,
            }
        )
        pbar.update()
    pbar.close()
    return results


def training_history(results: List[List[Dict[str, Any]]], model_id: int):
    histories = [r["history"] for r in results[model_id]]
    f, axs = plt.subplots(len(histories), 2, figsize=(7, 15), sharex=True)

    for i, history in enumerate(histories):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        acc = history.history["categorical_accuracy"]
        val_acc = history.history["val_categorical_accuracy"]
        actual_epochs = np.arange(1, len(loss) + 1)

        axs[i, 0].plot(actual_epochs, loss, label="loss")
        axs[i, 0].plot(actual_epochs, val_loss, label="val_loss")
        axs[i, 0].legend(prop={"size": 8})
        axs[i, 0].grid()
        axs[i, 0].set_ylabel("Entropia Cruzada")

        axs[i, 1].plot(actual_epochs, acc, label="loss")
        axs[i, 1].plot(actual_epochs, val_acc, label="val_loss")
        axs[i, 1].legend(prop={"size": 8})
        axs[i, 1].grid()
        axs[i, 1].axhline(1.0, color="green")
        axs[i, 1].set_ylabel("Exatidão")

    f.tight_layout()
    plt.show()
