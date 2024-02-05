import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow.keras import layers


def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    def __init__(self, latent_dim=4, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv_1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv_2 = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.fc_mean = layers.Dense(units=latent_dim)
        self.fc_log_var = layers.Dense(units=latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        with tf.name_scope("Encoder"):
            conv_1_o = self.conv_1(inputs)
            conv_2_o = self.conv_2(conv_1_o)
            # Flatten the convolution output
            x = tf.reshape(conv_2_o, [tf.shape(conv_2_o)[0], -1])
            z_mean = self.fc_mean(x)
            z_log_var = self.fc_log_var(x)
            z = self.sampling((z_mean, z_log_var))

            return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.fc_2 = layers.Dense(7 * 7 * 32, activation="relu")
        self.deconv_1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.deconv_2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        # No activation
        self.reconstructed = layers.Conv2DTranspose(1, 3, strides=1, padding="same")

    def call(self, inputs):
        with tf.name_scope("Decoder"):
            x = self.fc_2(inputs)
            x = tf.reshape(x, [tf.shape(x)[0], 7, 7, 32])
            deconv_1_o = self.deconv_1(x)
            deconv_2_o = self.deconv_2(deconv_1_o)
            return self.reconstructed(deconv_2_o)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        latent_dim=8,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        outputs = reconstructed, z_mean, z_log_var, z
        return outputs

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_loss(self, x, model_out):
        reconstructed, z_mean, z_log_var, z = model_out
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, z_mean, z_log_var)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def generate_and_save_images(model, epoch, test_sample):
  mean, logvar, z = model.encoder(test_sample)
  predictions = tf.sigmoid(model.decoder(z))
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('dump/test_vae_2/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


def plot_latent_distribution(model, dataset):
  for i, (x, y) in enumerate(dataset):
    # print('Current processing ', i)
    mean, logvar, z = model.encoder(x)
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

  plt.colorbar()
  plt.show()


def plot_latent_images(model, n, digit_size=28):
  """ Plots n x n digit images decoded from the latent space. """

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = tf.sigmoid(model.decoder(z))
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.show()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))
labelled_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_labels))).batch(32)


num_examples_to_generate = 16
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

optimizer = tf.keras.optimizers.Adam(1e-3)
epochs = 15
latent_dim = 2

vae = VariationalAutoEncoder(latent_dim=latent_dim)

generate_and_save_images(vae, 0, test_sample)

for epoch in range(epochs):
    start_time = time.time()
    running_loss = []
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            model_out = vae(x_batch_train)
            # Compute reconstruction loss & ELBO
            loss = vae.compute_loss(x_batch_train, model_out)
            running_loss.append(loss)
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    loss = tf.keras.metrics.Mean()
    for step, x_batch_test in enumerate(test_dataset):
        model_out = vae(x_batch_test)
        loss(vae.compute_loss(x_batch_test, model_out))
    elbo = -loss.result()
    end_time = time.time()

    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))

    generate_and_save_images(vae, epoch+1, test_sample)


# Save the model
vae.save('./checkpoints/trained_vae')

# Plot the latent image space -
plot_latent_images(vae, 20)

# Load the model
# vae = tf.keras.models.load_model('./checkpoints/trained_vae')
# vae.summary()

plot_latent_distribution(vae, labelled_dataset)


