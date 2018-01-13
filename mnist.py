from keras.models import Sequential, Model, Input
from keras.layers import Dense, Reshape, Flatten, Activation, Add
from keras.layers.convolutional import Convolution2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from blocks import HamLayer

from gan.wgan_gp import WGAN_GP
from gan.dataset import ArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
import keras.backend as K

import numpy as np

def make_generator():
    # model = Sequential()
    # model.add(Dense(1024, input_dim=128))
    # model.add(LeakyReLU())
    # model.add(Dense(128 * 8 * 8))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    # model.add(Reshape((8, 8, 128), input_shape=(128 * 7 * 7,)))
    # bn_axis = -1
    # model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Convolution2D(64, (5, 5), padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # # Because we normalized training inputs to lie in the range [-1, 1],
    # # the tanh function should be used for the output of the generator to ensure its output
    # # also lies in this range.
    # model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    # return model


    fwd = Sequential()
    fwd.add(UpSampling2D(size=(1, 8), data_format="channels_first", input_shape=(2, 2, 128)))
    #2x2x1024
    h0 = HamLayer(kernel_size=(1, 1))
    fwd.add(h0)
    #2x2x1024
    h1 = HamLayer(kernel_size=(1, 1))
    fwd.add(h1)
    #2x2x1024
    fwd.add(AveragePooling2D(pool_size=(1, 4), data_format='channels_first'))
    #fwd.add(UpSampling2D())
    #2x2x256
    h2 = HamLayer(h=1)
    fwd.add(h2)
    fwd.add(AveragePooling2D(pool_size=(1, 2), data_format='channels_first'))
    fwd.add(UpSampling2D())
    #4x4x128[0]
    h3 = HamLayer(h=1)
    fwd.add(h3)
    fwd.add(AveragePooling2D(pool_size=(1, 2), data_format='channels_first'))
    fwd.add(UpSampling2D())
    #8x8x64
    h4 = HamLayer(h=1)
    fwd.add(h4)
    #8x8x64
    h5 = HamLayer(h=1)
    fwd.add(h5)
    fwd.add(AveragePooling2D(pool_size=(1, 2), data_format='channels_first'))
    fwd.add(UpSampling2D())
    #16x16x32
    h6 = HamLayer(h=1)
    fwd.add(h6)
    #16x16x32
    h7 = HamLayer(h=1)
    fwd.add(h7)
    fwd.add(AveragePooling2D(pool_size=(1, 2), data_format='channels_first'))
    fwd.add(UpSampling2D())
    #32x32x16
    h8 = HamLayer(h=1)
    fwd.add(h8)
    fwd.add(AveragePooling2D(pool_size=(1, 16), data_format='channels_first'))


    bcw = Sequential()

    bcw.add(UpSampling2D(size=(1, 16), data_format='channels_first', input_shape=(32, 32, 1)))
    bcw.add(h8.get_backward())
    #32x32x16
    bcw.add(AveragePooling2D())
    bcw.add(UpSampling2D(size=(1, 2), data_format='channels_first'))
    bcw.add(h7.get_backward())
    #16x16x32
    bcw.add(h6.get_backward())
    #16x16x32
    bcw.add(AveragePooling2D())
    bcw.add(UpSampling2D(size=(1, 2), data_format='channels_first'))
    bcw.add(h5.get_backward())
    #8x8x64
    bcw.add(h4.get_backward())
    #8x8x64
    bcw.add(AveragePooling2D())
    bcw.add(UpSampling2D(size=(1, 2), data_format='channels_first'))
    bcw.add(h3.get_backward())
    #4x4x128
    bcw.add(AveragePooling2D())
    bcw.add(UpSampling2D(size=(1, 2), data_format='channels_first'))
    bcw.add(h2.get_backward())
    #2x2x256
    #bcw.add(AveragePooling2D())
    bcw.add(UpSampling2D(size=(1, 4), data_format='channels_first'))
    #2x2x1024
    bcw.add(h1.get_backward())
    #2x2x1024
    bcw.add(h0.get_backward())
    #2x2x128
    bcw.add(AveragePooling2D(pool_size=(1, 8), data_format='channels_first'))

    #32x32x1
    #model.add(Activation('tanh'))
    print (fwd.summary())
    print (bcw.summary())

    x = Input(batch_shape=(64, 32, 32, 1))
    z = Input(batch_shape=(64, 2, 2, 128))

    z_out = bcw(x)
    x_out = fwd(z)

    return fwd, bcw, Model([x, z], [x_out, z_out])


def make_discriminator():
    x = Input(batch_shape=(64, 32, 32, 1))
    z = Input(batch_shape=(64, 2, 2, 128))

    model = Sequential()
    model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(32, 32, 1)))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))

    model_fc = Sequential()
    model_fc.add(Convolution2D(256, (2, 2), padding='valid', input_shape=(2, 2, 128)))
    model_fc.add(Convolution2D(512, (1, 1)))
    model_fc.add(Convolution2D(1, (1, 1)))
    model_fc.add(Flatten())

    print (K.int_shape(model_fc(z)))

    out = Add()([model(x), model_fc(z)])

    return Model([x, z], out)

class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(2, 2, 128), bcw=None, fwd=None):
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate((X_train, X_test), axis=0)
        X_new = np.zeros((X.shape[0], 32, 32))
        X_new[:, :28, :28] = X
        X = X_new
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

        X = (X.astype(np.float32) - 127.5) / 127.5
        super(MNISTDataset, self).__init__(X, batch_size, noise_size)
        self.bcw = bcw
        self.fwd = fwd

    def next_generator_sample(self):
        index = self._next_data_index()
        return self._load_discriminator_data(index)

    def _load_discriminator_data(self, index):
        return [self._X[index], np.random.normal(size=(self._batch_size,) + self._noise_size)]

    def display(self, output_batch, input_batch = None):
        batch = output_batch[0]
        def batch_as_image(batch):
            image = super(MNISTDataset, self).display(batch)
            image = (image * 127.5) + 127.5
            image = np.clip(image, 0, 255)
            image = np.squeeze(np.round(image).astype(np.uint8))
            return image

        image = batch_as_image(batch)
        if (self.bcw is not None):
            data = self.next_discriminator_sample()[0]
            z = self.bcw.predict(data)
            rec = self.fwd.predict(z)
            data = batch_as_image(data)
            rec = batch_as_image(rec)
            return np.concatenate([image, data, rec], axis=1)
        else:
            return image



def main():
    fwd, bcw, generator = make_generator()
    discriminator = make_discriminator()

    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size, fwd=fwd, bcw=bcw)
    gan = WGAN_GP(generator,
                  discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
