from gan.wgan_gp import WGAN_GP, gradient_peanalty
from keras.models import Model, Input
from keras.optimizers import Adam
import keras.backend as K
from functools import partial

class MimGan(object):
    def __init__(self, fwd, bcw, discriminators,
                 generator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                 discriminator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                 gradient_penalty_weight=10, **kwargs):

        self.fwd = fwd
        self.bcw = bcw
        self.gradient_penalty_weight = gradient_penalty_weight
        self.discriminators = discriminators

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        num_outputs_fwd = len(self.fwd.output)
        num_outputs_bcw = len(self.bcw.output)

        self.fwd_input = self.fwd.input if type(self.fwd.input) == list else [self.fwd.input]
        self.bcw_input = self.bcw.input if type(self.bcw.input) == list else [self.bcw.input]

        assert (num_outputs_bcw == num_outputs_fwd)
        assert (num_outputs_fwd == len(self.discriminators))
        self.generator_metric_names = [str(i / 2) + '_' + ('fwd' if i % 2 == 0 else 'bcw')
                                       for i in range(num_outputs_fwd + num_outputs_bcw)]
        self.discriminator_metric_names = self.generator_metric_names + ['gp_' + str(i) for i in range(num_outputs_bcw)]


    def _set_trainable(self, net, trainable):
        for layer in net.layers:
            layer.trainable = trainable
        net.trainable = trainable

    def _compile_generator(self):
        self._set_trainable(self.bcw, True)
        self._set_trainable(self.fwd, True)
        [self._set_trainable(discriminator, False) for discriminator in self.discriminators]

        generator_model = Model(inputs=self.bcw_input + self.fwd_input, outputs=self.fwd_input)
        loss, metrics = self._compile_generator_loss()
        generator_model.compile(optimizer=self.generator_optimizer, loss=loss, metrics=metrics)
        return generator_model

    def _compile_generator_loss(self):
        def gen_loss_part(y_true, y_pred, index, is_fwd):
            return K.mean(-self.disc_out_fwd[index] if is_fwd else self.disc_out_bcw[index])

        losses = []
        for index in range(len(self.discriminators)):
            fn = partial(gen_loss_part, index=index, is_fwd=True)
            fn.__name__ = 'fwd_' + str(index)
            losses.append(fn)
            fn = partial(gen_loss_part, index=index, is_fwd=False)
            fn.__name__ = 'bcw_' + str(index)
            losses.append(fn)

        def gen_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), losses), K.zeros((1, )))

        return gen_loss, losses

    def _compile_discriminator(self):
        """
            Create model that produce discriminator scores from real_data and noise(that will be inputed to generator)
        """
        self._set_trainable(self.bcw, False)
        self._set_trainable(self.fwd, False)
        [self._set_trainable(discriminator, True) for discriminator in self.discriminators]

        discriminator_model = Model(inputs=self.bcw_input + self.fwd_input,
                                    outputs=self.fwd_input)
        loss, metrics = self._compile_discriminator_loss()
        discriminator_model.compile(optimizer=self.discriminator_optimizer, loss=loss, metrics=metrics)

        return discriminator_model

    def _compile_discriminator_loss(self):
        def disc_loss_part(y_true, y_pred, index, is_fwd):
            return K.mean(self.disc_out_fwd[index] if is_fwd else -self.disc_out_bcw[index])

        losses = []
        for index in range(len(self.discriminators)):
            fn = partial(disc_loss_part, index=index, is_fwd=True)
            fn.__name__ = 'fwd_' + str(index)
            losses.append(fn)
            fn = partial(disc_loss_part, index=index, is_fwd=False)
            fn.__name__ = 'bcw_' + str(index)
            losses.append(fn)

        gp_fn_losses = []
        for i, bcw, fwd, disc in zip(range(len(self.discriminators)), self.fwd_out,
                                     self.bcw_out, self.discriminators):
            gp_fn_list = gradient_peanalty([bcw], [fwd], self.gradient_penalty_weight, disc)
            gp_fn_list[0].__name__ = 'gp_loss' + str(i)
            gp_fn_losses.append(gp_fn_list[0])

        def gp_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), gp_fn_losses), K.zeros((1, )))

        def disc_loss(y_true, y_pred):
            return sum(map(lambda fn: fn(y_true, y_pred), losses), K.zeros((1, )))

        def disc_full_loss(y_true, y_pred):
            return gp_loss(y_true, y_pred) + disc_loss(y_true, y_pred)

        return disc_full_loss, losses + gp_fn_losses

    def get_generator(self):
        z = self.fwd_input
        x = self.bcw_input

        generator = Model(inputs=x + z, outputs=self.fwd(z))
        return generator

    def get_discriminator(self):
        disc_inp = [Input(K.int_shape(out)[1:]) for out in self.fwd_out]

        combined = Model(inputs=disc_inp,
                        outputs=[disc(inp) for inp, disc in zip(disc_inp, self.discriminators)])
        return combined

    def compile_models(self):
        self.fwd_out = self.fwd(self.fwd_input)
        self.bcw_out = self.bcw(self.bcw_input)

        self.disc_out_fwd = [disc(input) for disc, input in zip(self.discriminators, self.fwd_out)]
        self.disc_out_bcw = [disc(input) for disc, input in zip(self.discriminators, self.bcw_out)]

        return self._compile_generator(), self._compile_discriminator()
