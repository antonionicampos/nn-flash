import tensorflow as tf


class MLPCritic(tf.keras.Model):
    def __init__(self, hidden_units, activation="relu", *args, **kwargs):
        super(MLPCritic, self).__init__(*args, **kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.hidden_layers = [tf.keras.layers.Dense(units, activation=self.activation) for units in self.hidden_units]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        hidden = tf.identity(x)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)

    def get_config(self):
        return {"hidden_units": self.hidden_layers, "activation": self.activation}
