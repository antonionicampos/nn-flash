import tensorflow as tf


class MLPGenerator(tf.keras.Model):
    def __init__(self, output_dim, hidden_units, activation="relu", *args, **kwargs):
        super(MLPGenerator, self).__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.hidden_layers = [tf.keras.layers.Dense(units, activation=self.activation) for units in self.hidden_units]
        self.output_layer = tf.keras.layers.Dense(self.output_dim, activation="softmax", name="output")

    def call(self, x):
        hidden = tf.identity(x)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)

    def get_config(self):
        return {"output_dim": self.output_dim, "hidden_units": self.hidden_units, "activation": self.activation}
