import tensorflow as tf
import numpy as np
import autoencoder.model as autoencoder
import loadData
import params

lr = params.LEARNING_RATE


def train_input_fn():

    dataset_model = loadData.trainingSet(params.TRAININGSET_PATH)
    dataset = dataset_model['dataset']
    return dataset


def ae_fc_model(features, labels, mode, params):
    ae = autoencoder.build_full_connection_ae(is_weight_decay=params['is_weight_decay'], features=features)
    training_error = ae['training_error']
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(training_error, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=training_error, train_op=train_op)

def main(argv):
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=5 * 60,
        keep_checkpoint_max=5
    )
    ae = tf.estimator.Estimator(
        model_fn=ae_fc_model,
        params={
            'is_weight_decay': True
        },
        model_dir='./outputs/best',
        config=my_checkpointing_config
    )

    ae.train(input_fn=train_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
