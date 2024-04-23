import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from classification.mlp.models import MLPClassifier
from tqdm.notebook import tqdm

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
    data[["P", "T"]] = normalize(data[["P", "T"]])
    features = data[FEATURES_NAMES].copy()
    labels = pd.get_dummies(data["class"], dtype=np.float32)
    return features, labels


def training_model(train_files, valid_files, **kwargs):
    learning_rate = 0.001
    epochs = 500
    batch_size = 32

    pbar = tqdm(total=len(train_files))
    results = []

    for train_f, valid_f in zip(train_files, valid_files):
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
            tf.keras.callbacks.EarlyStopping(patience=10),
        ]
        model = MLPClassifier(**kwargs)
        model.compile(optimizer=optimizer, loss=loss_object, metrics=[accuracy])
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=callbacks,
            verbose=0,
        )
        results.append((model, history))
        description = train_f.split("\\")[-1].split(".")[0]
        pbar.set_description(description)
        pbar.update()

    return results


def training_history(results):
    histories = [r[1] for r in results]
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
        axs[i, 0].set_ylabel("Cross Entropy")

        axs[i, 1].plot(actual_epochs, acc, label="loss")
        axs[i, 1].plot(actual_epochs, val_acc, label="val_loss")
        axs[i, 1].legend(prop={"size": 8})
        axs[i, 1].grid()
        axs[i, 1].axhline(1.0, color="green")
        axs[i, 1].set_ylabel("Accuracy")

    f.tight_layout()
    plt.show()


def performance_evaluation(results, valid_files):
    m1 = tf.keras.metrics.CategoricalAccuracy()
    m2 = tf.keras.metrics.F1Score()

    accuracy = np.empty(shape=(10, len(results)))
    f1_score = np.empty(shape=(10, 3, len(results)))

    for i, res in enumerate(results):
        models = [r[0] for r in res]

        for j, (model, valid_f) in enumerate(zip(models, valid_files)):
            valid_data = pd.read_csv(valid_f)
            valid_features, valid_labels = preprocessing(valid_data)

            logits = model(valid_features.values)
            probs = tf.nn.softmax(logits)
            m1.update_state(valid_labels.values, probs)
            m2.update_state(valid_labels.values, probs)

            accuracy[j, i] = m1.result().numpy()
            f1_score[j, :, i] = m2.result().numpy().reshape(1, -1)

    return {"accuracy": accuracy, "f1_score": f1_score}
