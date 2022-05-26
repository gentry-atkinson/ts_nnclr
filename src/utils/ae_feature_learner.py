#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
#Build and trained an unsupervised feature extractor using
#  a convolutional autoencoder

import tensorflow as tf
from gen_ts_data import generate_pattern_data_as_array

latent_dim = 64
kernel_size = 16
output_size = 4
input_size = (128,128) 

encoder = tf.keras.models.Sequential()
encoder.add(tf.keras.layers.Input(input_size))
encoder.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu'))
encoder.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu'))
encoder.add(tf.keras.layers.MaxPooling1D(kernel_size))
encoder.add(tf.keras.layers.Flatten())
encoder.add(tf.keras.layers.Dense(latent_dim))

print(encoder.output.shape[1:])

decoder = tf.keras.models.Sequential()
decoder.add(tf.keras.layers.Input(input_shape=(latent_dim)))
decoder.add(tf.keras.layers.Reshape((1,latent_dim)))
decoder.add(tf.keras.layers.Conv1DTranspose(64, kernel_size, activation='relu'))
decoder.add(tf.keras.layers.Conv1DTranspose(64, kernel_size, activation='relu'))
decoder.add(tf.keras.layers.MaxPooling1D(kernel_size))
decoder.add(tf.keras.layers.Flatten())
decoder.add(tf.keras.layers.Dense(output_size, activation='linear'))

conv_autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
conv_autoencoder.compile(optimizer='adam', loss='mse')


def get_features_for_set(X):
    pass

if __name__ == '__main__':
  print('Verifying AutoEncoder')
  X = [
    generate_pattern_data_as_array(128) for _ in range(128)
  ]
  
  conv_autoencoder.summary()
  history = conv_autoencoder.fit(X, X, batch_size=64, epochs=40, shuffle=True, validation_split=0.1)
  