from keras.layers import Layer, AvgPool2D, UpSampling2D, Activation
import keras.initializers
import keras.backend as K

from keras.models import Model, Input
import numpy as np
from keras.backend import tf

assert K.image_data_format() == 'channels_last'


def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]

        new_shape = K.cast(new_shape, 'float64')
        new_shape *= tf.constant(np.array([height_factor, width_factor]))
        new_shape = K.cast(new_shape, 'int32')

        x = K.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]

        new_shape = K.cast(new_shape, 'float64')
        new_shape *= tf.constant(np.array([height_factor, width_factor]))
        new_shape = K.cast(new_shape, 'int32')

        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)


class HamLayer(Layer):
    def __init__(self, h=1, kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(HamLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.h = h

    def build(self, input_shape):
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        assert input_dim % 2 == 0
        kernel_shape = self.kernel_size + (input_dim / 2, input_dim / 2)
        self.filters = input_dim

        self.kernel_1 = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel_1')
        self.bias_1 = self.add_weight(shape=(self.filters / 2,),
                                        initializer=self.bias_initializer,
                                        name='bias_1')

        self.kernel_2 = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel_2')
        self.bias_2 = self.add_weight(shape=(self.filters / 2,),
                                        initializer=self.bias_initializer,
                                        name='bias_2')

        self.built = True


    def get_backward(self):
        return HamLayerBackward(self)

    def compute_output_shape(self, input_shape):
        return input_shape

    def split_input(self, x):
        return x[..., :(self.filters/2)], x[..., (self.filters/2):]
        #return x[..., ::2], x[..., 1::2]

    def merge_output(self, y, z):
        return K.concatenate([y, z], axis=-1)
        # y = K.expand_dims(y, axis=-1)
        # z = K.expand_dims(z, axis=-1)
        # x = K.concatenate([y, z], axis=-1)
        # z_shape = K.shape(z)
        # return K.reshape(x, (z_shape[0], z_shape[1], z_shape[2], -1))

    def branch(self, z, kernel, bias):
        out = K.conv2d(z, kernel=kernel, padding='same', strides=(1, 1))
        out = K.bias_add(out, bias)
        out = K.relu(out)
        z_shape = K.shape(z)
        tr_shape = (z_shape[0], z_shape[1], z_shape[2], self.filters / 2)
        out = K.conv2d_transpose(out, output_shape=tr_shape,
                                 kernel=kernel, padding='same', strides=(1, 1))
        return self.h * out

    # def resize_features(self, x):
    #     print (self.num_filters_mul)
    #     print (K.int_shape(x))
    #     x = resize_images(x, height_factor=1, width_factor=self.num_filters_mul, data_format='channels_first')
    #     print(K.int_shape(x))
    #     return x

    def call(self, inputs, **kwargs):
        x = inputs
        # x = self.resize_features(x)
        # #print (K.int_shape(x))
        y, z = self.split_input(x)
        y_new = y + self.branch(z, self.kernel_1, self.bias_1)
        z_new = z - self.branch(y_new, self.kernel_2, self.bias_2)
        return self.merge_output(y_new, z_new)

    def get_config(self):
        config = {"kernel_size": self.kernel_size, "h": self.h}
        base_config = super(HamLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HamLayerBackward(HamLayer):
    def __init__(self, ham_layer, **kwargs):
        super(HamLayerBackward, self).__init__(h = ham_layer.h, kernel_size=ham_layer.kernel_size,
                                               name=ham_layer.name + "_backward", **kwargs)
        self.ham_layer = ham_layer

    def build(self, input_shape):
        self.kernel_1 = self.ham_layer.kernel_1
        self.bias_1 = self.ham_layer.bias_1
        self.kernel_2 = self.ham_layer.kernel_2
        self.bias_2 = self.ham_layer.bias_2

        self.filters = self.ham_layer.filters
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        x = inputs
        y_new, z_new = self.split_input(x)
        z = z_new + self.branch(y_new, self.kernel_2, self.bias_2)
        y = y_new - self.branch(z, self.kernel_1, self.bias_1)
        x = self.merge_output(y, z)
        # x = self.resize_features(x)
        return x


class InvTanh(Layer):
    def call(self, inputs, **kwargs):
        return 0.5 * (K.log(1 + inputs + 1e-6) - K.log(1 - inputs + 1e-6))


def pool_updim_block(input, pool_size=(2, 2), dim_upsize = 2):
    out = AvgPool2D(input, pool_size)(input)
    out = UpSampling2D(size = (1, dim_upsize), data_format='channels_first')(out)
    return out

def upsize_downdim_block(input, upsize=(2, 2), dim_downsize = 2):
    out = AvgPool2D(input, pool_size=(1, dim_downsize), strides=(1, dim_downsize), data_format='channels_first')(input)
    out = UpSampling2D(size=upsize)(out)
    return out

def main():
    inp = Input((3, 3, 2))
    ham = HamLayer()
    bcw = ham.get_backward()

    mid = ham(inp)
    out = bcw(mid)

    m = Model(inp, out)
    m.summary()
    print (m.predict(np.arange(18).reshape(1, 3, 3, 2)))

if __name__ == "__main__":
    main()
