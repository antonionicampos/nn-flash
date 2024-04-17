import tensorflow as tf


class MLPCritic(tf.keras.Model):
    def __init__(self, hidden_units, activation="relu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units,
                activation=activation,
            )
            for units in hidden_units
        ]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        hidden = tf.identity(x)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)


class DeepConvCritic(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        hidden = tf.identity(x)

        # Convolution 1 (64 filters) + LeakyReLU + Dropout
        hidden = self.conv1(hidden)
        hidden = self.leaky_relu(hidden)

        # Convolution 2 (128 filters) + LeakyReLU + Dropout
        hidden = self.conv2(hidden)
        hidden = self.batch_norm1(hidden)
        hidden = self.leaky_relu(hidden)

        # Convolution 2 (256 filters) + LeakyReLU + Dropout
        hidden = self.conv3(hidden)
        hidden = self.batch_norm2(hidden)
        hidden = self.leaky_relu(hidden)

        hidden = self.flatten(hidden)
        return self.dense(hidden)
