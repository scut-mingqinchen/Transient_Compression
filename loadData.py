import h5py
import time
import random
import params
import numpy as np
import os
import tensorflow as tf

np.set_printoptions(threshold=np.inf)


class generator:
    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            perm = [i for i in range(np.int32(params.N_FILE_SAMPE / params.BATCH_SIZE))]
            random.shuffle(perm)
            dataset = hf['/training_data']
            for i in range(np.int32(params.N_FILE_SAMPE / params.BATCH_SIZE)):
                data = dataset[perm[i] * params.BATCH_SIZE:(perm[i] + 1) * params.BATCH_SIZE]
                yield data


def trainingSet(dir):
    filenames = os.listdir(dir)
    filenames = [dir + i for i in filenames]
    ds = tf.data.Dataset.from_tensor_slices(filenames).shuffle(40)
    ds = ds.apply(tf.contrib.data.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
        generator(),
        tf.float32,
        tf.TensorShape([params.BATCH_SIZE, params.N_INPUT_FEATURES, params.N_PATCH_SIZE, params.N_PATCH_SIZE, 1]),
        args=(filename,)), cycle_length=4, sloppy=True)).repeat(
        50).prefetch(2)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    model = {'dataset': ds,
             'iterator': iterator,
             'next_element': next_element
             }

    return model


#########################################################
# Our normalization function
#########################################################
def log_norm(input):
    output = input.copy()
    output[input > 0] = np.log10(input[input > 0]) + 7
    output[input < 1e-7] = 0
    return output


def de_log_norm(input):
    output = input.copy()
    output[input > 0] = np.power(10, input[input > 0] - 7)
    return output
