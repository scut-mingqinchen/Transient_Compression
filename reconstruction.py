import autoencoder.inference as inference
import tensorflow as tf
import h5py
import params
import numpy as np
import loadData


def reconstruct_full_video(modeldir, input, savepath):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ###################################################################################
        # Due to GPU memory, only reconstruct 30 x 1 x 1 x 4096 once
        ###################################################################################
        params.BATCH_SIZE = 30
        model = inference.get_model()
        saver = model['saver']
        saver.restore(sess, modeldir)
        file = h5py.File(input, "r")
        data = file['/xyt'][()]
        padding = np.zeros([309, 309, 4096])
        padding[4:304, 4:304, :] = data
        recon_data = np.zeros([300, 300, 4096])
        for i in range(300):
            x = i + 4
            test_list = []
            for j in range(300):
                y = j + 4
                test = np.reshape(np.transpose(padding[x - 4:x + 5, y - 4:y + 5, :], (2, 1, 0)),
                                  [4096, 9, 9, 1])
                test_list.append(test)
                if np.mod(j + 1, 30) == 0:
                    test_list = np.array(test_list)

                    #########################################################
                    # recon --- reconstructed volume (batchsize,4096)
                    # code  --- encoder output (batchsize,128)
                    # loss  --- loss function value
                    #########################################################
                    recon, code, loss = inference.infer_ae_session(data=loadData.log_norm(test_list),
                                                                   sess=sess,
                                                                   model=model)
                    recon_data[i, j - 29:j + 1, :] = recon
                    test_list = []
        recon_data = loadData.de_log_norm(recon_data)
        file = h5py.File(savepath, "w")
        dataset = file.create_dataset("xyt", (300, 300, 4096), dtype='float32', compression="gzip")
        dataset[:, :, :] = np.asarray(recon_data, 'float32')


def demo_recon_video():
    #########################################################
    # Reconstruct one video
    # modeldir --- path of trained model
    # input --- full video data
    # savepath --- path of reconstructed video
    #########################################################
    modeldir = './trained_model/model.ckpt-406418'
    input = './data/church_albedo_1_view_2.h5'
    savepath = './outputs/church_albedo_1_view_2_ours2.h5'
    reconstruct_full_video(modeldir, input, savepath)


def demo_recon_anydata():

    #########################################################
    # Prepare data and trained model
    #########################################################
    params.BATCH_SIZE = 30
    input = np.random.random([params.BATCH_SIZE, params.N_INPUT_FEATURES, params.N_PATCH_SIZE, params.N_PATCH_SIZE, 1])
    modeldir = './trained_model/model.ckpt-406418'

    #########################################################
    # Normalizing the data
    #########################################################
    norm_input=loadData.log_norm(input)

    #########################################################
    # Infer the data through the network
    # recon --- reconstructed data (batchsize,4096)
    # code  --- encoder output (batchsize,128)
    # loss  --- loss function value
    #########################################################
    recon, code, loss=inference.infer_ae(norm_input,modeldir)

    #########################################################
    # Denormalizing the data
    #########################################################
    recon_data = loadData.de_log_norm(recon)

    #########################################################
    # Visit the data
    #########################################################
    print(np.shape(recon_data))

if __name__ == '__main__':
    demo_recon_video()
    #demo_recon_anydata()