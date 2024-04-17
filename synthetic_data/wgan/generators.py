import tensorflow as tf


class MLPGenerator(tf.keras.Model):
    def __init__(self, output_dim, hidden_units, activation="relu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation=activation)
            for units in hidden_units
        ]
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation="softmax",
            name="output",
        )

    def call(self, x):
        hidden = tf.identity(x)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)


class DeepConvGenerator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense = tf.keras.layers.Dense(4 * 4 * 4 * 64, use_bias=False)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.reshape_layer = tf.keras.layers.Reshape((4, 4, 4 * 64))

        self.upsampling1 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.upsampling2 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.upsampling3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            activation="sigmoid",
        )

        self.crop2d = tf.keras.layers.Cropping2D(cropping=(2, 2))

    def call(self, x):
        hidden = tf.identity(x)

        hidden = self.dense(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.relu(hidden)

        hidden = self.reshape_layer(hidden)

        hidden = self.upsampling1(hidden)
        hidden = self.batch_norm1(hidden)
        hidden = self.relu(hidden)

        hidden = self.upsampling2(hidden)
        hidden = self.batch_norm2(hidden)
        hidden = self.relu(hidden)

        return self.crop2d(self.upsampling3(hidden))
