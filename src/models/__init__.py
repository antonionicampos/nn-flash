import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVC
from src.utils import denorm
from src.utils.constants import REGRESSION_TARGET_NAMES, VAPOR_FRACTIONS_NAMES, LIQUID_FRACTIONS_NAMES

np.set_printoptions(suppress=True)


class FlashModel:
    def __init__(
        self,
        stability_model: tf.keras.Model | SVC,
        phase_model: tf.keras.Model,
        targets_min: np.ndarray,
        targets_max: np.ndarray,
    ):
        self.stability_model = stability_model
        self.phase_model = phase_model
        self.targets_min = targets_min
        self.targets_max = targets_max

    def __call__(self, inputs: pd.DataFrame):
        X = tf.convert_to_tensor(inputs)

        # Stability test
        logits = self.stability_model(X)
        probs = tf.nn.softmax(logits, axis=1)
        pred_label = tf.argmax(probs, axis=1).numpy()

        # Phase Split
        # Get mixtures
        mixture_samples = np.argwhere(pred_label == 1).flatten()
        X_mix = tf.convert_to_tensor(inputs.iloc[mixture_samples, :])
        mix_outputs = denorm(self.phase_model(X_mix).numpy(), self.targets_min, self.targets_max)
        mix_outputs = pd.DataFrame(mix_outputs, columns=REGRESSION_TARGET_NAMES)
        Z = inputs.iloc[mixture_samples, inputs.columns.str.startswith("z")].to_numpy()
        mix_outputs = self.convert_K_to_fractions(Z, mix_outputs)
        mix_outputs.index = mixture_samples

        # Get liquid/vapor
        # vapor (gas)
        comp_size = len(LIQUID_FRACTIONS_NAMES)
        vap_samples = np.argwhere(pred_label == 0).flatten()
        vap_comp = pd.DataFrame(inputs.iloc[vap_samples, :-2].values, columns=VAPOR_FRACTIONS_NAMES)
        liq_comp = pd.DataFrame(np.zeros((vap_samples.shape[0], comp_size)), columns=LIQUID_FRACTIONS_NAMES)
        vap_outputs = pd.concat((liq_comp, vap_comp), axis=1)
        vap_outputs.loc[:, "nV"] = 1.0
        vap_outputs.index = vap_samples

        # liquid (oil)
        comp_size = len(VAPOR_FRACTIONS_NAMES)
        liq_samples = np.argwhere(pred_label == 2).flatten()
        liq_comp = pd.DataFrame(inputs.iloc[liq_samples, :-2].values, columns=LIQUID_FRACTIONS_NAMES)
        vap_comp = pd.DataFrame(np.zeros((liq_samples.shape[0], comp_size)), columns=VAPOR_FRACTIONS_NAMES)
        liq_outputs = pd.concat((liq_comp, vap_comp), axis=1)
        liq_outputs.loc[:, "nV"] = 0.0
        liq_outputs.index = liq_samples

        return pd.concat((mix_outputs, vap_outputs, liq_outputs)).sort_index()

    def convert_K_to_fractions(self, Z, outputs):
        K = outputs.loc[:, outputs.columns.str.startswith("K_")].to_numpy()
        V = outputs.loc[:, ["nV"]].to_numpy()
        L = 1 - V
        X = Z / (L + V * K)
        Y = K * X

        X = pd.DataFrame(X, columns=LIQUID_FRACTIONS_NAMES)
        Y = pd.DataFrame(Y, columns=VAPOR_FRACTIONS_NAMES)

        return pd.concat([X, Y, outputs.loc[:, "nV"]], axis=1)
