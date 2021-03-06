from gan.inception_score import get_inception_score
from gan.fid import calculate_fid_given_arrays
import numpy as np
from tqdm import tqdm
import keras.backend as K


def compute_scores(epoch, image_shape, generator, dataset, number_of_images=50000, compute_inception=True, compute_fid=True,
                   log_file=None, cache_file='mnist_fid.npz', additional_info=""):
    if not (compute_inception or compute_fid):
        return
    images = np.empty((number_of_images, ) + image_shape)

    generator_input = generator.get_input_at(0)
    if type(generator_input) != list:
        generator_input = [generator_input]

    generator_output = generator.get_output_at(0)
    if type(generator_output) != list:
        generator_output = [generator_output]

    predict_fn = K.function(generator_input + [K.learning_phase()], [generator_output[0]])

    for begin in tqdm(range(0, number_of_images, dataset._batch_size)):
        end = min(number_of_images, begin + dataset._batch_size)
        n_images = end - begin
        g_s = dataset.next_generator_sample_test()
        images[begin:end] = predict_fn(g_s + [False])[0][:n_images]

    images *= 127.5
    images += 127.5

    def to_rgb(array):
        if array.shape[-1] != 3:
            #hack for grayscale mnist
            return np.concatenate([array, array, array], axis=-1)
        else:
            return array

    if compute_inception:
        str = "INCEPTION SCORE: %s, %s" % get_inception_score(to_rgb(images))
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s " % (epoch, )) + str + " " + additional_info
    if compute_fid:
        true_images = 127.5 * dataset._X_test + 127.5
        str = "FID SCORE: %s" % calculate_fid_given_arrays([to_rgb(true_images)[:number_of_images],
                                                            to_rgb(images)[:number_of_images]], cache_file=cache_file)
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s " % (epoch, )) + str + " " + additional_info
