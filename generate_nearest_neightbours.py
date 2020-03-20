import sys
import numpy as np
np.random.seed(2)
import tensorflow as tf
tf.random.set_random_seed(2)
import pickle
import os
from models.rae import make_raes
from models.std_vae import std_vae
from models.wae_mmd import wae_mmd
from dataloaders.dataloader import DataLoader
from configurations import config
from my_utility import config_parser
import keras.backend as K
from my_utility import accumulate_batches_of_data_frm_generator
from sklearn.neighbors import KDTree
from scipy.misc import imsave


def main():

    # Setting up logging
    maj_cfg_idx, minor_cfg_idx = config_parser.get_config_idxs(int(sys.argv[1]), config.configurations)
    log_root = config.configurations[maj_cfg_idx][0]['log_root']
    log_root = os.path.join(log_root, str(maj_cfg_idx))
    log_dir = os.path.join(log_root, config.configurations[maj_cfg_idx][minor_cfg_idx]['expt_name'] +\
                                     '_' + str(minor_cfg_idx))

    model_name = config.configurations[maj_cfg_idx][0]['base_model_name'] + '_' + \
                 config.configurations[maj_cfg_idx][0]['dataset_name'] + '.h5'
    model_path = os.path.join(log_dir, model_name)

    expt_name = config.configurations[maj_cfg_idx][minor_cfg_idx]['expt_name']
    n_components = config.configurations[maj_cfg_idx][minor_cfg_idx]['n_components']

    # Preparing data Generator
    batch_size = config.configurations[maj_cfg_idx][0]['batch_size']
    dataloader = DataLoader(batch_size=batch_size)
    (train_generator, validation_generator, test_generator), input_shape, (train_steps, val_steps, test_steps) = \
        dataloader.get_data_loader(dataset_name=config.configurations[maj_cfg_idx][0]['dataset_name'], shuffle=False)

    # Preparing model
    if config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('REDUCED') >= 0:
        encoder, decoder, auto_encoder = make_raes.get_vae(input_shape, config.configurations, maj_cfg_idx,
                                                           minor_cfg_idx)
        compute_z_cov = True
    elif config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('WAE') >= 0:
        encoder, decoder, auto_encoder = wae_mmd.get_wae(input_shape, config.configurations, maj_cfg_idx,
                                                         minor_cfg_idx)
        compute_z_cov = False
    elif config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('STD_VAE') >= 0:
        encoder, decoder, auto_encoder = std_vae.get_vae(input_shape, config.configurations, maj_cfg_idx,
                                                         minor_cfg_idx)
        compute_z_cov = False
    else:
        raise NotImplementedError("No implemntation for " +
                                  str(config.configurations[maj_cfg_idx][0]['base_model_name']) + " found.")

    multi_output_enc = False
    if len(encoder.outputs) > 1:
        multi_output_enc = True

    # Generatig sampled, reconstructed and interpolated images
    batches = 100
    # auto_encoder.load_weights(model_path + '_model_weights.h5')
    auto_encoder.load_weights(model_path+'_best')
    # auto_encoder.load_weights(model_path+'_model_weights.h5')

    np.random.seed(2)
    tf.random.set_random_seed(2)

    training_images = accumulate_batches_of_data_frm_generator.get_n_batches_of_input(train_steps, train_generator)
    training_images_shape = training_images.shape

    tree = KDTree(np.reshape(training_images, [training_images_shape[0], -1]), leaf_size=2)

    num_samples = 20
    num_neighbours = 3

    qz_est_name_list = ['N_0_I', 'GMM_1', 'GMM_10', 'GMM_20', 'GMM_100', 'GMM_200']
    for method_name in qz_est_name_list:
        if method_name == 'N_0_I':
            sampled_images = decoder.predict(np.random.normal(loc=0.0, scale=1.0,
                                                           size=(num_samples,
                                                                 K.get_variable_shape(encoder.outputs[0])[1])))
        else:
            with open(os.path.join(log_dir, method_name+'_mdl.pkl'), 'rb') as f:
                gmm = pickle.load(f)

            sampled_images = decoder.predict(gmm.sample(num_samples)[0])

        for smpld_img_idx in range(sampled_images.shape[0]):
            _, neighbour_inds = tree.query(np.reshape(sampled_images[smpld_img_idx].flatten(), (1, -1)),
                                           k=num_neighbours)
            img_together = np.squeeze(np.hstack([np.hstack(training_images[neighbour_inds[0]]), sampled_images[smpld_img_idx]]))
            save_dir = os.path.join(log_dir, method_name+"_neighbours")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            imsave(os.path.join(save_dir, str(smpld_img_idx)+'.png'), img_together)


if __name__ == "__main__":
    main()