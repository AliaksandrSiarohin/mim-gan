from keras.models import Sequential, Model, Input, clone_model
from keras.layers import Dense, Reshape, Flatten, Activation, Add, Lambda, BatchNormalization
from keras.layers.convolutional import Convolution2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

from blocks import HamLayer, SpaceToDepth, DepthToSpace, Upsample, Downsample

from gan.dataset import ArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
import keras.backend as K
from gdan import GDAN
from gan.gan import GAN
import numpy as np

def make_generator_branch(z, name='generator'):
    out = Dense(1024)(z)
    out = LeakyReLU()(out)
    out = Dense(128 * 7 * 7)(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = Reshape((7, 7, 128))(out)
    out = Conv2DTranspose(128, (5, 5), strides=2, padding='same')(out)
    out = BatchNormalization()(out)
    out = Convolution2D(64, (5, 5), padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2DTranspose(64, (5, 5), strides=2, padding='same')(out)
    out = BatchNormalization()(out)
    out = Convolution2D(1, (5, 5), padding='same', activation='tanh')(out)
    return out

def make_generator():

    z = Input((128, ))
    out1 = make_generator_branch(z, name='generator')
    return Model(z, out1)


def make_discriminator():
    x = Input((28, 28, 1))
    out = Convolution2D(64, (5, 5), padding='same')(x)
    out = LeakyReLU()(out)
    out = Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2])(out)
    out = LeakyReLU()(out)
    out = Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2])(out)
    out = LeakyReLU()(out)
    out = Flatten()(out)
    out = Dense(1024, kernel_initializer='he_normal')(out)
    out = LeakyReLU()(out)
    out = Dense(1, kernel_initializer='he_normal')(out)
    return Model(x, out)

class MNISTDataset(ArrayDataset):
    def __init__(self, batch_size, noise_size=(128, )):
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X = (X.astype(np.float32) - 127.5) / 127.5
        super(MNISTDataset, self).__init__(X, batch_size, noise_size)

    def number_of_batches_per_epoch(self):
        return 100

    def display(self, output_batch, input_batch = None):
        batch = output_batch[0]
        def batch_as_image(batch):
            image = super(MNISTDataset, self).display(batch)
            image = (image * 127.5) + 127.5
            image = np.clip(image, 0, 255)
            image = np.squeeze(np.round(image).astype(np.uint8))
            return image

        image = batch_as_image(batch)
        return image
        #img_gd = batch_as_image(output_batch[1])
        #return np.concatenate([image, img_gd], axis=1)


def main():
    generator = make_generator()
    discriminator = make_discriminator()
    generator.summary()
    discriminator.summary()

    args = parser_with_default_args().parse_args()
    dataset = MNISTDataset(args.batch_size)
    gan = GDAN(generator=generator, discriminator=discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    trainer.train()

if __name__ == "__main__":
    main()
