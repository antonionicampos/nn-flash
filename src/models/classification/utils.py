import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product


def load_model_hparams(hparameters):
    models_hparams = []
    for hp in hparameters:
        if hp.get("model_type", "") == "svm":
            svm_hparams = []
            # For 'poly' kernel
            params = dict(hp["params"])
            params["kernel"] = ["poly"]
            keys, values = list(params.keys()), list(params.values())
            svm_hparams += [{keys[i]: params[i] for i in range(len(params))} for params in list(product(*values))]

            # For other kernels
            params = dict(hp["params"])
            params.pop("degree")
            keys, values = list(params.keys()), list(params.values())
            svm_hparams += [{keys[i]: params[i] for i in range(len(params))} for params in list(product(*values))]

            models_hparams += [
                {"model_name": f"SVM #{i+1}", "model_type": hp["model_type"], "params": {**svm_hp}}
                for i, svm_hp in enumerate(svm_hparams)
            ]
        elif hp.get("model_type", "") == "neural_network":
            if "model_name" in hp:
                models_hparams += [dict(hp)]
            else:
                nn_hparams = []
                nn_params = dict(hp["params"])
                hidden_units = []
                for num_layers in nn_params["hidden_layers"]:
                    if num_layers == 0:
                        hidden_units += [[]]
                    else:
                        for num_units in nn_params["hidden_units"]:
                            hidden_units += [[num_units] * num_layers]

                nn_params.pop("hidden_layers")
                nn_params.pop("hidden_units")

                keys, values = ["hidden_units"] + list(nn_params.keys()), [hidden_units] + list(nn_params.values())
                nn_hparams += [{keys[i]: params[i] for i in range(len(params))} for params in list(product(*values))]
                models_hparams += [
                    {
                        "model_name": f"Rede Neural #{i+1}",
                        "model_type": hp["model_type"],
                        "params": {**nn_hp},
                        "opt": hp["opt"],
                    }
                    for i, nn_hp in enumerate(nn_hparams)
                ]

    return models_hparams


def model_parameters_size(model: tf.keras.Model):
    parameters = [params.numpy().flatten().shape[0] for params in model.trainable_variables]
    return np.prod(parameters)


def binary_classification(model: tf.keras.Model, data: pd.DataFrame, label: int):
    features, labels = data["features"], data["targets"]
    X = tf.convert_to_tensor(features)
    y = tf.convert_to_tensor(labels)
    y = tf.argmax(y, axis=1)

    logits = model(X)
    probs = tf.nn.softmax(logits)
    y_class = np.where(y == label, 1, 0)
    y_hat_class = probs[:, label]
    return y_class, y_hat_class
