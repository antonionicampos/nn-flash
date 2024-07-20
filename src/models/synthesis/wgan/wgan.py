import tensorflow as tf

from typing import Callable, Dict


class WGAN(tf.keras.Model):
    """Wasserstein GAN
    https://arxiv.org/abs/1701.07875
    """

    def __init__(
        self,
        critic: tf.keras.Model,
        generator: tf.keras.Model,
        latent_dim: int,
        n_critic: int = 5,
        c: float = 0.01,
    ):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.c = c

    def compile(
        self,
        critic_optimizer: tf.keras.optimizers.RMSprop,
        generator_optimizer: tf.keras.optimizers.RMSprop,
        critic_loss_fn: Callable[[tf.Tensor, tf.Tensor], float],
        generator_loss_fn: Callable[[tf.Tensor], float],
    ):
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_loss_fn = critic_loss_fn
        self.generator_loss_fn = generator_loss_fn

    def train_step(self, real_samples: tf.Tensor):
        batch_size = tf.shape(real_samples)[0]
        for _ in range(self.n_critic):
            random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                generated_samples = self.generator(random_latent_vectors)
                generated_logits = self.critic(generated_samples)
                real_logits = self.critic(real_samples)

                critic_loss = self.critic_loss_fn(real_logits, generated_logits)

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

            # weight clipping of the critic model to enforce Lipschitz constraint
            for var in self.critic.trainable_variables:
                var.assign(tf.clip_by_value(var, -self.c, self.c))

        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors)
            generated_samples_logits = self.critic(generated_samples)

            generator_loss = self.generator_loss_fn(generated_samples_logits)

        generator_grad = tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_grad, self.generator.trainable_variables)
        )
        return {"critic_loss": critic_loss, "generator_loss": generator_loss}


class WGANGP(tf.keras.Model):
    """Wasserstein GAN with Gradient Penalty
    https://arxiv.org/abs/1704.00028
    """

    def __init__(
        self,
        critic: tf.keras.Model,
        generator: tf.keras.Model,
        latent_dim: int,
        n_critic: int = 5,
        lambda_: float = 10.0,
    ):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_ = lambda_

    def compile(
        self,
        critic_optimizer: tf.keras.optimizers.Adam,
        generator_optimizer: tf.keras.optimizers.Adam,
        critic_loss_fn: Callable[[tf.Tensor, tf.Tensor], float],
        generator_loss_fn: Callable[[tf.Tensor], float],
    ) -> None:
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_loss_fn = critic_loss_fn
        self.generator_loss_fn = generator_loss_fn

    def gradient_penalty(
        self, batch_size: int, real_samples: tf.Tensor, generated_samples: tf.Tensor
    ) -> tf.Tensor:
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated samples
        num_axis = len(real_samples.shape)
        axis = [batch_size] + [1 for _ in range(num_axis - 1)]

        random_number = tf.random.uniform(axis, dtype=tf.float32)
        interpolated = (
            random_number * real_samples + (1 - random_number) * generated_samples
        )

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # Get the critic output for this interpolated image.
            pred = self.critic(interpolated)

        # Calculate the gradients w.r.t to this interpolated samples.
        grads = tape.gradient(pred, [interpolated])[0]

        # Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        grad_penalty = tf.reduce_mean((norm - 1) ** 2)

        return grad_penalty

    def train_step(self, real_samples: tf.Tensor) -> Dict[str, float]:
        batch_size = tf.shape(real_samples)[0]

        for _ in range(self.n_critic):
            # Get latent vectors
            random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                generated_samples = self.generator(random_latent_vectors)
                generated_logits = self.critic(generated_samples)
                real_logits = self.critic(real_samples)

                critic_loss = self.critic_loss_fn(real_logits, generated_logits)
                gp = self.gradient_penalty(batch_size, real_samples, generated_samples)
                critic_loss += gp * self.lambda_

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

        # Train the generator
        # Get latent vectors
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors)
            generated_samples_logits = self.critic(generated_samples)

            generator_loss = self.generator_loss_fn(generated_samples_logits)

        generator_grad = tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_grad, self.generator.trainable_variables)
        )

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}


def critic_loss(real_logits: tf.Tensor, generated_logits: tf.Tensor) -> float:
    real_loss = tf.reduce_mean(real_logits)
    generated_loss = tf.reduce_mean(generated_logits)
    return generated_loss - real_loss


def generator_loss(generated_logits: tf.Tensor) -> float:
    return -tf.reduce_mean(generated_logits)
