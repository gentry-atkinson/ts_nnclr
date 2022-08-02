#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
#Build and trained an unsupervised feature extractor using
#  a convolutional autoencoder

from gc import callbacks
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
#from utils.gen_ts_data import generate_pattern_data_as_array

latent_dim = 32
kernel_size = 16
input_size = (128,1) 

def make_encoder():
  encoder = tf.keras.models.Sequential()
  encoder.add(tf.keras.layers.Input(input_size))
  encoder.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu', data_format='channels_last'))
  encoder.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu'))
  encoder.add(tf.keras.layers.MaxPooling1D(kernel_size))
  encoder.add(tf.keras.layers.Flatten())
  encoder.add(tf.keras.layers.Dense(latent_dim, activation='relu'))

  return encoder

def make_decoder(encoder):
  decoder = tf.keras.models.Sequential()
  decoder.add(tf.keras.layers.Input(encoder.output.shape[1:]))
  decoder.add(tf.keras.layers.Reshape((latent_dim,1)))
  decoder.add(tf.keras.layers.Conv1DTranspose(64, kernel_size, activation='relu', data_format='channels_last'))
  decoder.add(tf.keras.layers.Conv1DTranspose(64, kernel_size, activation='relu'))
  decoder.add(tf.keras.layers.MaxPooling1D(kernel_size))
  decoder.add(tf.keras.layers.Flatten())
  decoder.add(tf.keras.layers.Dense(input_size[0]*input_size[1], activation='linear'))
  decoder.add(tf.keras.layers.Reshape(input_size))

  return decoder

def make_cae():
  encoder = make_encoder()
  decoder = make_decoder(encoder)
  conv_autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
  conv_autoencoder.compile(optimizer='adam', loss='mse')

  return conv_autoencoder


def get_features_for_set(X, with_visual=False, with_summary=False, returnModel=False):
    global latent_dim
    global kernel_size
    global input_size
    input_size = X[0].shape

    print("AE input shape: ", input_size)
    
    if len(X[0]) <= 32:
      kernel_size = 4
    elif len(X[0]) <= 128:
      kernel_size = 8

    ae = make_cae()
    if with_summary: ae.summary()

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,verbose=1)

    history = ae.fit(X, X, batch_size=16, epochs=100, shuffle=True, validation_split=0.1, callbacks=es)
    if with_visual:
      plt.figure()
      reconstructed_signal = ae.predict(X)[0]
      plt.plot(range(len(X[0])), X[0])
      plt.plot(range(len(X[0])), reconstructed_signal)
      plt.show()

    feature_encoder = tf.keras.models.Sequential(ae.layers[:-1])
    if with_summary: feature_encoder.summary()

    if returnModel:
      return feature_encoder
    else:
      return feature_encoder.predict(X)

if __name__ == '__main__':
  print('Verifying AutoEncoder')
  # X = np.array([
  #   generate_pattern_data_as_array(128) for _ in range(128)
  # ])
  
  # X = np.reshape(X, (128,128,1))
  # print(X.shape)

  # encoded_X = get_features_for_set(X)