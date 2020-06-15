import tensorflow as tf
import params
import numpy as np

encode_patch_size = [9, 9, 7, 7, 5, 3, 1]
decode_patch_size = [1, 1, 1, 1, 1, 1, 1]
encode_features = [4096, 2048, 1024, 512, 256, 128, 128]
encode_channel = [1, 64, 64, 128, 64, 64, 1]
decode_features = encode_features[::-1]
decode_channel = encode_channel[::-1]

list_stride = [1, 2, 1, 1, 1]
pad = 'SAME'


def build_full_connection_ae(is_trainable=True,
                             is_weight_decay=False, features=None, is_dropout=True):
    n_layer_encoder = len(encode_features)
    n_layer_decoder = len(decode_features)

    #########################################################
    # Set placeholders
    #########################################################
    if features is None:
        x_data_in_node = tf.placeholder(params.TF_DATA_TYPE,
                                        [params.BATCH_SIZE, params.N_INPUT_FEATURES, params.N_PATCH_SIZE,
                                         params.N_PATCH_SIZE, 1])
    else:
        x_data_in_node = features

    label = x_data_in_node[:, :, 4, 4, 0]
    response = x_data_in_node
    ae_weight_list = []

    #########################################################
    # Add noise in the center pixel
    #########################################################
    if is_dropout:
        sita = 1. - params.DROP_RATE
        mask = np.ones([params.BATCH_SIZE, 4096, 9, 9, 1], dtype='float32')
        mask[:, :, 4, 4, 0] = 0
        response = tf.multiply(mask, response) + tf.pad(tf.nn.dropout(response[:, :, 4:5, 4:5, :], sita) * sita,
                                                        [[0, 0], [0, 0], [4, 4], [4, 4], [0, 0]])

    #########################################################
    # Build the encoder
    #########################################################
    layer_name_base = 'encoder'
    for l in range(len(encode_features) - 1):

        layer_name = layer_name_base + '-layer' + str(l)

        n_feature_prev = encode_features[l]
        n_feature_next = encode_features[l + 1]

        n_channel_prev = encode_channel[l]
        n_channel_next = encode_channel[l + 1]

        with tf.variable_scope(layer_name):
            if l == 0 or l == 2:
                conv_weight = tf.get_variable("weight",
                                              shape=[3, 3, 3, n_channel_prev, n_channel_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d(response, conv_weight, list_stride, "SAME")
            elif l == n_layer_encoder - 2:
                conv_weight = tf.get_variable("weight",
                                              shape=[1, 3, 3, n_channel_prev, n_channel_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d(response, conv_weight, [1, 1, 1, 1, 1], "VALID")
            else:
                conv_weight = tf.get_variable("weight",
                                              shape=[2, 3, 3, n_channel_prev, n_channel_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d(response, conv_weight, list_stride, "VALID")

            conv_bias = tf.Variable(tf.zeros([n_channel_next], dtype=params.TF_DATA_TYPE),
                                    name='bias',
                                    trainable=is_trainable)

            response = tf.nn.bias_add(conv, conv_bias)
            response = tf.layers.batch_normalization(response)
            response = tf.nn.tanh(response)
            if is_weight_decay:
                ae_weight_list.append(conv_weight)

    # ########################################################
    # define code
    # ########################################################
    response_code = tf.nn.relu(tf.squeeze(response))

    #########################################################
    # Build the decoder
    #########################################################
    layer_name_base = 'decoder'

    for l in range(len(decode_features) - 1):

        layer_name = layer_name_base + '-layer' + str(l)

        n_feature_prev = decode_features[l]
        n_feature_next = decode_features[l + 1]

        n_channel_prev = decode_channel[l]
        n_channel_next = decode_channel[l + 1]

        n_patch_size_prev = decode_patch_size[l]
        n_patch_size_next = decode_patch_size[l + 1]

        with tf.variable_scope(layer_name):
            if l == 0:
                conv_weight = tf.get_variable("weight",
                                              shape=[3, 1, 1, n_channel_next, n_channel_prev],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d_transpose(response, conv_weight,
                                              [params.BATCH_SIZE, n_feature_next, n_patch_size_next, n_patch_size_next,
                                               n_channel_next],
                                              [1, 1, 1, 1, 1], "SAME")

                conv_bias = tf.Variable(tf.zeros([n_channel_next], dtype=params.TF_DATA_TYPE),
                                        name='bias',
                                        trainable=is_trainable)
            else:
                conv_weight = tf.get_variable("weight",
                                              shape=[2, 1, 1, n_channel_next, n_channel_prev],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d_transpose(response, conv_weight,
                                              [params.BATCH_SIZE, n_feature_next, n_patch_size_next, n_patch_size_next,
                                               n_channel_next],
                                              list_stride, "VALID")

                conv_bias = tf.Variable(tf.zeros([n_channel_next], dtype=params.TF_DATA_TYPE),
                                        name='bias',
                                        trainable=is_trainable)
            response = tf.nn.bias_add(conv, conv_bias)
            response = tf.layers.batch_normalization(response)
            response = tf.nn.tanh(response)
            if is_weight_decay:
                ae_weight_list.append(conv_weight)

            #########################################################
            # There are three convolution layers after deconvolution
            #########################################################
            for i in range(3):
                conv_weight = tf.get_variable("weight_conv" + str(i),
                                              shape=[3, 1, 1, n_channel_next, n_channel_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable, dtype=params.TF_DATA_TYPE)
                conv = tf.nn.conv3d(response, conv_weight, [1, 1, 1, 1, 1], "SAME")

                conv_bias = tf.Variable(tf.zeros([n_channel_next], dtype=params.TF_DATA_TYPE),
                                        name='bias' + str(i),
                                        trainable=is_trainable)

                response = tf.nn.bias_add(conv, conv_bias)
                response = tf.layers.batch_normalization(response)

                ############################################################
                # residual connection and the last activate function is Relu
                ############################################################
                if i == 2:
                    response = rnn_response + response
                if l == n_layer_decoder - 2 and i == 2:
                    response = tf.nn.relu(response)
                else:
                    response = tf.nn.tanh(response)
                if i == 0:
                    rnn_response = response

                if is_weight_decay:
                    ae_weight_list.append(conv_weight)

    response = tf.squeeze(response)

    #########################################################
    # define output
    #########################################################
    x_data_out_node = tf.nn.relu(response)

    #########################################################
    # define the loss - data term
    #########################################################
    data_loss = tf.reduce_mean(tf.square(tf.pow(2.0, response) - tf.pow(2.0, label)),
                               name='training_error')
    training_error = data_loss

    #########################################################
    # define the loss - weight decay term
    #########################################################
    if is_weight_decay:
        weight_lambda = params.TF_WEIGHT_DECAY_LAMBDA
        weight_decay_term = weight_lambda * tf.add_n([tf.nn.l2_loss(v) for v in ae_weight_list])
        weight_decay_term /= len(ae_weight_list)
        training_error += weight_decay_term

    #########################################################
    # Add summaries
    #########################################################
    tf.summary.scalar('training error', training_error)
    tf.summary.scalar('data loss', data_loss)

    if is_weight_decay:
        tf.summary.scalar('regularization loss', weight_decay_term)

    merged = tf.summary.merge_all()

    #########################################################
    # Add saver
    #########################################################
    saver = tf.train.Saver(max_to_keep=3)

    #########################################################
    # Return model
    #########################################################
    model = {'x_data_in_node': x_data_in_node,
             'x_data_out_node': x_data_out_node,
             'code': response_code,
             'training_error': training_error,
             'data_loss': data_loss,
             'saver': saver,
             'summary': merged,
             'ae_weight_list': ae_weight_list,
             }

    if is_weight_decay:
        model['weight_decay_term'] = weight_decay_term

    return model
