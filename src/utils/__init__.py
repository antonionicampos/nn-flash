import neqsim
import numpy as np
import tensorflow as tf

from itertools import product
from src.utils.constants import NEQSIM_COMPONENTS
from typing import Dict


def load_model_hparams(hparameters):
    models_hparams = []
    for hp in hparameters:
        if hp["model_type"] == "svm":
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
        elif hp["model_type"] == "neural_network" or hp["model_type"] == "wgan":
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
                        "model_name": (
                            f"Rede Neural #{i+1}" if hp["model_type"] == "neural_network" else f"WGAN #{i+1}"
                        ),
                        "model_type": hp["model_type"],
                        "params": {**nn_hp},
                        "opt": hp["opt"],
                    }
                    for i, nn_hp in enumerate(nn_hparams)
                ]
        elif hp["model_type"] == "dirichlet":
            models_hparams += [dict(hp)]

    return [{"model_id": i + 1, **hp} for i, hp in enumerate(models_hparams)]


def model_parameters_size(model: tf.keras.Model):
    parameters = [params.numpy().flatten().shape[0] for params in model.trainable_variables]
    return np.prod(parameters)


def create_fluid(composition: Dict[str, float]):
    """Create NeqSim fluid adding its components and fractions.

    Parameters
    ----------
    composition : Dict[str, float]
        fluid components with its fractions

    Returns
    -------
        NeqSim fluid.
    """
    fluid = neqsim.thermo.fluid("pr")  # Peng-Robinson EOS
    for component, fraction in composition.items():
        fluid.addComponent(NEQSIM_COMPONENTS[component], fraction)
    fluid.setMixingRule("classic")  # classic will use binary kij
    return fluid


def denorm(data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray):
    return data * (max_vals - min_vals) + min_vals
