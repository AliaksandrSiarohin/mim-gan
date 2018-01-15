from keras.models import Sequential, Model, Input
from keras.layers import Dense, Reshape, Flatten, Activation, Add
from keras.layers.convolutional import Convolution2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

from blocks import HamLayer, SpaceToDepth, DepthToSpace, Upsample, Downsample

from gan.dataset import ArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
import keras.backend as K
from min_gan import MimGan

import numpy as np


def make_generator():
    x = Input((32, 32, 1))
    z = Input((2, 2, 16))

    layers = []

    def ham_block(input, kernel_size=(3, 3), upsample=True, depth_to_space=False):
        h = HamLayer(kernel_size=kernel_size)
        layers.append(h)
        out = h(input)
        if upsample:
            if depth_to_space:
                upl = DepthToSpace()
                layers.append(upl)
                out = upl(out)
            else:
                upl = Upsample()
                avg = Downsample(pool_size=(1, 2), data_format='channels_first')
                layers.append(avg)
                layers.append(upl)
                out = upl(out)
                out = avg(out)
        return out

    fwd_out = z
    fwd_out = Reshape((1, 1, 64))(fwd_out)
    l = Upsample(size=(1, 16), data_format='channels_first')
    layers.append(l)
    fwd_out = l(fwd_out)
    fwd_out = ham_block(fwd_out, kernel_size=(1, 1), upsample=False)
    fwd_out = ham_block(fwd_out, kernel_size=(1, 1), upsample=True)
    #2x2x256
    fwd_out = ham_block(fwd_out, kernel_size=(2, 2), upsample=False, depth_to_space=True)
    #2x2x256
    fwd_out = ham_block(fwd_out, kernel_size=(2, 2), upsample=True, depth_to_space=True)
    #4x4x64
    fwd_out = ham_block(fwd_out, upsample=False)
    #4x4x64
    fwd_out = ham_block(fwd_out, upsample=True, depth_to_space=True)
    #8x8x16
    fwd_out = ham_block(fwd_out, upsample=False)
    #8x8x16
    fwd_out = ham_block(fwd_out, upsample=True)
    #16x16x4
    fwd_out = ham_block(fwd_out, upsample=False)
    #16x16x4
    fwd_out = ham_block(fwd_out, upsample=True)
    #32x32x4
    fwd_out = ham_block(fwd_out, upsample=False)
    l = Downsample(pool_size=(1, 8), data_format='channels_first')
    layers.append(l)
    fwd_out = l(fwd_out)

    fwd = Model(inputs=z, outputs=[fwd_out, z])
    fwd.summary()

    bcw_out = x
    for l in layers[::-1]:
        bcw_out = l.get_backward()(bcw_out)
    bcw_out = Reshape((2, 2, 16))(bcw_out)
    bcw = Model(inputs=x, outputs=[x, bcw_out])

    return fwd, bcw, fwd


def make_discriminators():
    x = Input((32, 32, 1))
    z = Input((2, 2, 16))

    d1 = Sequential(name='disc_image')
    d1.add(Convolution2D(64, (5, 5), padding='same', input_shape=K.int_shape(x)[1:]))
    d1.add(LeakyReLU())
    d1.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    d1.add(LeakyReLU())
    d1.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    d1.add(LeakyReLU())
    d1.add(Flatten())
    d1.add(Dense(1024, kernel_initializer='he_normal'))
    d1.add(LeakyReLU())
    d1.add(Dense(1, kernel_initializer='he_normal'))

    d2 = Sequential()
    d2.add(Convolution2D(1024, kernel_size=(2, 2), input_shape=K.int_shape(z)[1:], activation='relu'))
    d2.add(Flatten())
    d2.add(Dense(1024, activation='relu'))
    d2.add(Dense(1))

    return [d1, d2]

class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(2, 2, 16), bcw=None, fwd=None):
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
        return [self._X[index], np.random.normal(size=(self._batch_size,) + self._noise_size)]

    def _load_discriminator_data(self, index):
        return []

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
            data = self.next_generator_sample()[0]
            z = self.bcw.predict(data)[1]
            rec = self.fwd.predict(z)[0]
            data = batch_as_image(data)
            rec = batch_as_image(rec)
            return np.concatenate([image, data, rec], axis=1)
        else:
            return image



def main():
    fwd, bcw, generator = make_generator()
    discriminators = make_discriminators()

    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size, fwd=fwd, bcw=bcw)
    gan = MimGan(fwd=fwd, bcw=bcw, discriminators=discriminators, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
