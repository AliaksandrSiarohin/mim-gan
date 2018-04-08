from gan.gan import GAN
from keras.backend import tf as ktf
from keras.layers import Input
import keras.backend as K
from keras.models import clone_model

class GDAN(GAN):
    def __init__(self, gradient_penalty_weight_generator=1, **kwargs):
        self.gradient_penalty_weight_generator = gradient_penalty_weight_generator

        super(GDAN, self).__init__(**kwargs)
        inp = self.generator_input[0]
        inp = Input(name='inp', tensor=inp)
        self.grad_generator = clone_model(self.generator, input_tensors=inp)
        #self.grad_generator self.generator_input
        self.grad_generator_output = self.grad_generator(self.generator_input)

    def compile_intermediate_variables(self):
        self.generator_output = self.generator(self.generator_input)
        #self.grad_generator_output = generator_output[1]
        self.discriminator_fake_output = self.discriminator(self.generator_output)
        self.discriminator_real_output = self.discriminator(self.discriminator_input)

    def get_gradient_penalty_loss(self, for_discriminator=True):
        if self.gradient_penalty_weight == 0:
            return []

        inp = self.discriminator_input if for_discriminator else self.generator_input
        if type(inp) == list:
            batch_size = ktf.shape(inp[0])[0]
        else:
            batch_size = ktf.shape(inp)[0]

        points = self.grad_generator_output
        print K.int_shape(points)

        gp_list = []
        disc_out = self.discriminator([points])
        if type(disc_out) != list:
            disc_out = [disc_out]
        gradients = ktf.gradients(disc_out[0], points)

        for gradient in gradients:
            if gradient is None:
                continue
            gradient = ktf.reshape(gradient, (batch_size, -1))
            gradient_l2_norm = ktf.sqrt(ktf.reduce_sum(ktf.square(gradient), axis=1))
            if for_discriminator:
                gradient_penalty = self.gradient_penalty_weight * ktf.square(1 - gradient_l2_norm)
            else:
                gradient_penalty = -self.gradient_penalty_weight_generator * gradient_l2_norm
            gp_list.append(ktf.reduce_mean(gradient_penalty))

        if for_discriminator:
            for i in range(len(gp_list)):
                self.discriminator_metric_names.append('gp_loss_' + str(i))
        return gp_list

    def compile_generator_train_op(self):
        loss_list = []
        adversarial_loss = self.get_generator_adversarial_loss(self.generator_adversarial_objective)
        loss_list.append(adversarial_loss)

        loss_list += self.additional_generator_losses()
        self.generator_loss_list = loss_list

        updates = self.generator_optimizer.get_updates(params=self.generator.trainable_weights + self.grad_generator.trainable_weights,
                                                       loss=sum(loss_list))
        updates += self.collect_updates(self.generator)
        updates += self.collect_updates(self.grad_generator)
        print (self.collect_updates(self.generator))

        lr_update = (self.lr_decay_schedule_generator(self.generator_optimizer.iterations) *
                                K.get_value(self.generator_optimizer.lr))
        updates.append(K.update(self.generator_optimizer.lr, lr_update))

        train_op = K.function(self.generator_input + self.additional_inputs_for_generator_train + [K.learning_phase()],
                            [sum(loss_list)] + loss_list, updates=updates)
        return train_op


    def additional_generator_losses(self):
        grad_loss = self.get_gradient_penalty_loss(for_discriminator=False)
        self.generator_metric_names.append('grad')
        return grad_loss

    def compile_generate_op(self):
        return K.function(self.generator_input + self.additional_inputs_for_generator_train + [K.learning_phase()],
                        self.generator_output + [self.grad_generator_output])
