import tensorflow as tf


class NeuralNetClassifier(tf.keras.Model):
    def __init__(self, hidden_units=[32], activation="relu", *args, **kwargs):
        super(NeuralNetClassifier, self).__init__(*args, **kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.hiddens = [
            tf.keras.layers.Dense(units, activation=self.activation, name=f"dense{i+1}")
            for i, units in enumerate(self.hidden_units)
        ]
        self.outputs = tf.keras.layers.Dense(3, name="output")

    def call(self, inputs):
        for hidden in self.hiddens:
            inputs = hidden(inputs)
        return self.outputs(inputs)

    def get_config(self):
        return {"hidden_units": self.hidden_units, "activation": self.activation}

    # def predict(self, inputs, output_type, **kwargs):
    #     output_types = ["logit", "probs", "label"]
    #     assert output_type in output_types, f"'output_type' parameter must be 'logit', 'probs' or 'label'."
    #     outputs = self(inputs)
    #     if output_type == "probs":
    #         outputs = tf.nn.softmax(outputs)
    #     elif output_type == "label":
    #         outputs = tf.nn.softmax(outputs)
    #         outputs = tf.argmax(outputs, axis=1)
    #     return outputs
