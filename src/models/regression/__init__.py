import tensorflow as tf


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
