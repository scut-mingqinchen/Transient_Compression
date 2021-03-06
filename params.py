import tensorflow as tf
import numpy as np

TF_DATA_TYPE = tf.float32
TF_WEIGHT_DECAY_LAMBDA = 5e-6
BATCH_SIZE = 25
DROP_RATE = 0.3
N_INPUT_FEATURES =4096
N_SAMPLE = np.int32(6000)
N_PATCH_SIZE = 9
N_BATCH = np.int32(145 * N_SAMPLE / BATCH_SIZE)
LEARNING_RATE = 5e-5
N_FILE_SAMPE = np.int32(21750)
TRAININGSET_PATH = "F:/9x9_6000sample/"

