import tensorflow as tf


class CustomHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {"batch": [], "critic_loss": []}

    def on_batch_end(self, batch, logs={}):
        self.history["batch"].append(batch)
        self.history["critic_loss"].append(logs.get("critic_loss"))
