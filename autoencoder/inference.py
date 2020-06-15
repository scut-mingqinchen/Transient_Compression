import tensorflow as tf
import autoencoder.model as ae_model


def get_model(is_weight_decay=False):
    return ae_model.build_full_connection_ae(is_weight_decay=is_weight_decay)


def infer_ae_session(data, sess, model):
    x_data_in_node = model['x_data_in_node']
    x_data_out_node = model['x_data_out_node']
    data_loss = model['data_loss']
    ae_codes = model['code']

    recon_list = []
    feed_dict = {x_data_in_node: data}
    recon, code, loss = sess.run([x_data_out_node, ae_codes, data_loss], feed_dict=feed_dict)
    recon_list.append(recon)

    return recon, code, loss


def infer_ae(data, filename_model, is_weight_decay=False):
    #########################################################
    # Generate the model
    #########################################################
    model = ae_model.build_full_connection_ae(is_weight_decay=is_weight_decay)

    x_data_in_node = model['x_data_in_node']
    x_data_out_node = model['x_data_out_node']
    data_loss = model['data_loss']
    ae_codes = model['code']
    saver = model['saver']

    #########################################################
    # Perform inference
    #########################################################
    # with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    with tf.Session() as sess:
        saver.restore(sess, filename_model)
        print("Model restored from file: %s" % filename_model)
        feed_dict = {x_data_in_node: data}
        recon, code, loss = sess.run([x_data_out_node, ae_codes, data_loss], feed_dict=feed_dict)
    tf.reset_default_graph()
    return recon, code, loss
