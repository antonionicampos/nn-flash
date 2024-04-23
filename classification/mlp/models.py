import tensorflow as tf


class MLPClassifier(tf.keras.Model):
    def __init__(self, hidden_layers=[32], activation=tf.keras.activations.relu):
        super().__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation=activation, name="dense1")
            for units in hidden_layers
        ]
        self.output_layer = tf.keras.layers.Dense(3, name="output")

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
