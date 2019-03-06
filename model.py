
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, LSTM, RepeatVector, Lambda
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras import backend as K


def generator_model(inputdim = 100, xdim = 4, ydim = 4):
    # xdim = 2, ydim = 2 results in prediction shape of (1, 3, 32, 32)
    # xdim = 4, ydim = 4 results in prediction shape of (1, 3, 64, 64)
    model = Sequential()
    model.add(Dense(input_dim=inputdim, output_dim=1024*xdim*ydim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape( (1024, xdim, ydim), input_shape=(inputdim,) ) )
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 4, 4, border_mode='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(128, 4, 4, subsample=(2, 2), input_shape=(3, 64, 64), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 4, 4, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 4, 4, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(1024, 4, 4, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    #discriminator.trainable = False
    return model







