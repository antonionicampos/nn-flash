import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, hidden_units=[32], activation=tf.keras.activations.relu):
        super().__init__()
        self.hiddens = [
            tf.keras.layers.Dense(units, activation=activation, name=f"dense{i+1}")
            for i, units in enumerate(hidden_units)
        ]
        self.output = tf.keras.layers.Dense(25, name="output")

    def call(self, x):
        for layer in self.hiddens:
            x = layer(x)
        return self.output(x)


class ResidualBlock(tf.keras.Model):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silu = tf.keras.activations.swish

        self.hidden1 = tf.keras.layers.Dense(units, activation=self.silu)
        self.hidden2 = tf.keras.layers.Dense(units)

    def call(self, x):
        inputs = tf.identity(x)
        h1 = self.hidden1(inputs)
        h2 = self.hidden2(h1)

        h2 += x
        return self.silu(h2)


class ResidualNeuralNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.hidden1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.keras.activations.swish,
            name="hidden1",
        )
        self.residual1 = ResidualBlock(64, name="residual1")
        self.residual2 = ResidualBlock(64, name="residual2")
        self.residual3 = ResidualBlock(64, name="residual3")
        self.concatenate = tf.keras.layers.Concatenate(name="concatenate")
        self.output = tf.keras.layers.Dense(25, name="output")

    def call(self, x):
        inputs = tf.identity(x)
        h1 = self.hidden1(inputs)
        r1 = self.residual1(h1)
        r2 = self.residual2(r1)
        r3 = self.residual3(r2)
        c1 = self.concatenate([r3, x])
        return self.output(c1)
