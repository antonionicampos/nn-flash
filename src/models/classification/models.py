import tensorflow as tf


class NeuralNetClassifier(tf.keras.Model):
    def __init__(self, hidden_units=[32], activation=tf.keras.activations.relu):
        super().__init__()
        self.hiddens = [
            tf.keras.layers.Dense(units, activation=activation, name=f"dense{i+1}")
            for i, units in enumerate(hidden_units)
        ]
        self.outputs = tf.keras.layers.Dense(3, name="output")

    def call(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return self.outputs(x)
