import tensorflow as tf

from src.utils import denorm


class MeanSquaredErrorWithSoftConstraint:

    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def __call__(self, y_true, y_pred, inputs, min_vals, max_vals):
        y_pred_denorm = tf.convert_to_tensor(denorm(y_pred, min_vals, max_vals), dtype=tf.float32)

        Khat = y_pred_denorm[:, :-1]
        nVhat = y_pred_denorm[:, -1:]

        xhat = inputs[:, :-2] / (1 + nVhat * (Khat - 1))
        yhat = Khat * xhat

        sum_xhat = tf.reduce_sum(xhat, axis=-1, keepdims=True)
        sum_yhat = tf.reduce_sum(yhat, axis=-1, keepdims=True)
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1, keepdims=True)

        loss = mse + self.lambda_ * tf.abs(sum_xhat + sum_yhat - 2)
        return tf.reduce_mean(loss)


class NeuralNet(tf.keras.Model):
    def __init__(self, hidden_units, activation, *args, **kwargs):
        super(NeuralNet, self).__init__(*args, **kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.hiddens = [
            tf.keras.layers.Dense(units, activation=self.activation, name=f"dense{i+1}")
            for i, units in enumerate(self.hidden_units)
        ]
        self.outputs = tf.keras.layers.Dense(25, activation="sigmoid", name="output")

    def call(self, inputs):
        for hidden in self.hiddens:
            inputs = hidden(inputs)
        return self.outputs(inputs)

    def get_config(self):
        return {"hidden_units": self.hidden_units, "activation": self.activation}


class ResidualBlock(tf.keras.Model):
    def __init__(self, units, *args, **kwargs):
        super(ResidualBlock, self).__init__(*args, **kwargs)
        self.units = units
        self.silu = tf.keras.activations.silu
        self.hidden1 = tf.keras.layers.Dense(units, activation="silu")
        self.hidden2 = tf.keras.layers.Dense(units)

    def call(self, inputs):
        x = tf.identity(inputs)
        h1 = self.hidden1(x)
        h2 = self.hidden2(h1)
        h2 += inputs
        return self.silu(h2)

    def get_config(self):
        return {"units": self.units, "silu": self.silu}


class ResidualNeuralNet(tf.keras.Model):
    def __init__(self, hidden_units, *args, **kwargs):
        super(ResidualNeuralNet, self).__init__(*args, **kwargs)
        self.hidden_units = hidden_units
        self.hidden1 = tf.keras.layers.Dense(units=64, activation="silu", name="hidden1")
        self.residuals = [ResidualBlock(units, name=f"residual{i+1}") for i, units in enumerate(self.hidden_units)]
        self.concatenate = tf.keras.layers.Concatenate(name="concatenate")
        self.outputs = tf.keras.layers.Dense(25, activation="sigmoid", name="output")

    def call(self, inputs):
        x = tf.identity(inputs)
        x = self.hidden1(x)
        for residual in self.residuals:
            x = residual(x)
        concat = self.concatenate([x, inputs])
        return self.outputs(concat)

    def get_config(self):
        return {"hidden_units": self.hidden_units}
