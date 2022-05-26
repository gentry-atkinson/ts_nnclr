#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
#Build and trained an unsupervised feature extractor using
#  a convolutional autoencoder

import tensorflow as tf

latent_dim = 64
kernel_size = 16 

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Conv1D(filters = 32, kernel_size=kernel_size),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        tf.keras.layers.Conv1DTranspose(filters = 32, kernel_size=kernel_size),
        tf.keras.layers.Dense(784, activation='linear'),
        tf.keras.layers.Flatten()
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

def get_features_for_set(X):
    global kernel_size
    if len(X[0]) < 100: kernel_size = 8
    elif len(X[0]) < 32: kernel_size = 4

    autoencoder.compile()
    autoencoder.fit(X, X, epochs =10, shuffle=True)
    return
