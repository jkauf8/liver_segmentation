import tensorflow as tf
from tensorflow import keras
from keras import backend as K

def conv_block(input, num_filters):
  x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
  
  return x

def encoder_block(input, num_filters):
  x = conv_block(input, num_filters)
  p = tf.keras.layers.MaxPooling2D((2, 2))(x)

  return x,p

def decoder_block(input, skip_features, num_filters):
  x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input)
  x = tf.keras.layers.concatenate([x, skip_features])
  x = conv_block(x,num_filters)
  
  return x


def unet(input_shape):
  inputs = tf.keras.layers.Input(input_shape)
  s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

  #encoder/contraction path
  c1, p1 = encoder_block(s, 16)
  c2, p2 = encoder_block(p1, 32)
  c3, p3 = encoder_block(p2, 64)
  c4, p4 = encoder_block(p3, 128)

  b1 = conv_block(p4, 256)

  #decoder/expansive path
  d1 = decoder_block(b1, c4, 128)
  d2 = decoder_block(d1, c3, 64)
  d3 = decoder_block(d2, c2, 32)
  d4 = decoder_block(d3, c1, 16)

  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  
  return model


