import tensorflow as tf


class MLPGenerator(tf.keras.Model):
    def __init__(self, output_dim, hidden_units, activation="relu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layers = [tf.keras.layers.Dense(units, activation=activation) for units in hidden_units]
        self.output_layer = tf.keras.layers.Dense(output_dim, activation="softmax", name="output")

    def call(self, x):
        hidden = tf.identity(x)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)
