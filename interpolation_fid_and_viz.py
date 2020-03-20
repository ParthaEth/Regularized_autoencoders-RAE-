
import sys
import numpy as np
np.random.seed(2)
import tensorflow as tf
# tf.random.set_random_seed(2)
sys.path.append('./precision_recall_distributions')
import os
from models.rae import make_raes
from models.std_vae import std_vae
from models.wae_mmd import wae_mmd
from dataloaders.dataloader import DataLoader
from configurations import config
from my_utility import config_parser
import keras.backend as K
from my_utility.my_callbacks import LatentSpaceSampler
from my_utility import save_batches_of_images
from my_utility import interpolations
from my_utility import fid_from_dir_computer
from my_utility import estimate_density_and_sample
import time
from precision_recall_distributions import prd_from_image_folders as prd


def predict_2stage(encoder, decoder, qz_sampler, recon_original):
    if len(encoder.outputs) > 1:
        return decoder.predict(qz_sampler.reconstruct(encoder.predict(recon_original)[0]))
    else:
        return decoder.predict(qz_sampler.reconstruct(encoder.predict(recon_original)))


pairs_interpolation = {'MNIST': [[537, 9749],
                                 [1327, 6570],
                                 [1703, 4717],
                                 [1838, 1399],
                                 [2028, 8637],
                                 [2543, 5672],
                                 [4118, 4817],
                                 [4471, 170],
                                 [4656, 8901],
                                 [5134, 2283],
                                 [5320, 912],
                                 [5676, 2381],
                                 [5977, 2686],
                                 [5983, 3868],
                                 [6816, 9143],
                                 [7409, 1415],
                                 [8027, 1636],
                                 [8739, 5640],
                                 [8960, 4306],
                                 [9316, 825]],
                       'CIFAR_10': [[537, 9749],
                                    [1327, 6570],
                                    [1703, 4717],
                                    [1838, 1399],
                                    [2028, 8637],
                                    [2543, 5672],
                                    [4118, 4817],
                                    [4471, 170],
                                    [4656, 8901],
                                    [5134, 2283],
                                    [5320, 912],
                                    [5676, 2381],
                                    [5977, 2686],
                                    [5983, 3868],
                                    [6816, 9143],
                                    [7409, 1415],
                                    [8027, 1636],
                                    [8739, 5640],
                                    [8960, 4306],
                                    [9316, 825]],
                       'CELEBA': [[190, 1526],
                                  [526, 15140],
                                  [1185, 1384],
                                  [3328, 9392],
                                  [5832, 8602],
                                  [7674, 10954],
                                  [8481, 787],
                                  [8765, 127],
                                  [9230, 11958],
                                  [10572, 16050],
                                  [10856, 12309],
                                  [11047, 1344],
                                  [11228, 11558],
                                  [11388, 14825],
                                  [11487, 17382],
                                  [13806, 6168],
                                  [15064, 15036],
                                  [15798, 14732],
                                  [17953, 7791],
                                  [18488, 16407],]}


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
    if config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('RAE') >= 0:
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

    # # # Compute precission and recall
    # prd.compute_prd(reference_dir=os.path.join(log_dir, 'recon_original'),
    #                 eval_dirs=[os.path.join(log_dir, 'one_gaussian_sampled'),
    #                            os.path.join(log_dir, 'GMM_10_sampled'),],
    #                 inception_path='/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/'
    #                                'frozen_inception_v1_2015_12_05/inceptionv1_for_inception_score.pb')
    #

    multi_output_enc = False
    if len(encoder.outputs) > 1:
        multi_output_enc = True

    # Generatig sampled, reconstructed and interpolated images
    batches = 100
    # auto_encoder.load_weights(model_path + '_model_weights.h5')
    auto_encoder.load_weights(model_path+'_best')
    # auto_encoder.load_weights(model_path+'_model_weights.h5')

    # Save embeddings
    train_labels = dataloader.get_train_labels()
    z_dims = K.get_variable_shape(encoder.outputs[0])[-1]

    # ## Training embeddings
    if not os.path.exists(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_training_embedding.npz')):
        zs_trn = np.zeros(((train_steps-1)*batch_size, z_dims))
        zs_trn_log_sigma = np.zeros(((train_steps - 1) * batch_size, z_dims))
        for i in range(train_steps-1):
            (x, _) = train_generator.next()
            if multi_output_enc:
                zs_trn[i * batch_size:(i + 1) * batch_size], zs_trn_log_sigma[i * batch_size:(i + 1) * batch_size] = \
                    encoder.predict(x)
            else:
                zs_trn [i*batch_size:(i+1)*batch_size] = encoder.predict(x)
        np.savez(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_training_embedding.npz'),
                 zs=zs_trn, labels=train_labels[:zs_trn.shape[0]])
        np.savez(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_training_embedding_log_sig.npz'),
                 zs_trn_log_sigma=zs_trn_log_sigma)
    else:
        zs_trn = np.load(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_training_embedding.npz'))['zs']
    #

    ## Validation embeddings
    if not os.path.exists(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_valid_embedding.npz')):
        valid_labels = dataloader.get_validation_labels()
        valid_embd = np.zeros(((val_steps - 1) * batch_size, z_dims))
        for i in range(val_steps - 1):
            (x, _) = validation_generator.next()
            if multi_output_enc:
                valid_embd[i * batch_size:(i + 1) * batch_size] = encoder.predict(x)[0]
            else:
                valid_embd[i * batch_size:(i + 1) * batch_size] = encoder.predict(x)
        np.savez(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_valid_embedding.npz'), zs=valid_embd,
                 labels=valid_labels[:valid_embd.shape[0]])
    else:
        valid_embd = np.load(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_valid_embedding.npz'))['zs']


    ## Test embeddings
    if test_generator is not None:
        if not os.path.exists(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_test_embedding.npz')):
            test_labels = dataloader.get_test_labels()
            zs_test = np.zeros(((test_steps - 1) * batch_size, z_dims))
            for i in range(test_steps - 1):
                (x, _) = test_generator.next()
                if multi_output_enc:
                    zs_test[i * batch_size:(i + 1) * batch_size] = encoder.predict(x)[0]
                else:
                    zs_test[i * batch_size:(i + 1) * batch_size] = encoder.predict(x)
            np.savez(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_test_embedding.npz'), zs=zs_test,
                     labels=test_labels[:zs_test.shape[0]])
        else:
            zs_test = np.load(os.path.join(log_dir, model_name[:-2] + '_' + expt_name + '_' + '_test_embedding.npz'))['zs']

    latent_sapce_sampler = LatentSpaceSampler(encoder, compute_z_cov=compute_z_cov)

    # # the following training_images are for training latentspace variance it doesn't matter much any more. Dont panic!
    # if test_generator is None:
    #     training_images = get_n_batches_of_input(batches, train_generator)
    # else:
    #     training_images = get_n_batches_of_input(batches, validation_generator)
    #
    # if config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('REDUCED') < 0:
    #     # For any model other than ours zs should be sampled from std normal. The following flag forces that
    #     latent_sapce_sampler.multi_output_encoder = True
    #
    # zs = latent_sapce_sampler.get_zs(training_images)

    np.random.seed(2)
    tf.random.set_random_seed(2)
    tf.compat.v1.random.set_random_seed(2)
    pairs = np.random.choice(list(range(valid_embd.shape[0]))*3, (10000, 2), replace=False)

    dataset_dir = dataloader.get_data_dir()
    # ###
    # # # # # save sampled images
    sampled_images = decoder.predict(np.random.normal(loc=0.0, scale=1.0, size=(10000, zs_test.shape[-1])))
    save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'one_gaussian_sampled'), sampled_images)
    print("FID of random samples using N(0, I) is : " +
          str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                            dataset_dir,
                                            os.path.join(log_dir, 'one_gaussian_sampled'))))
    # ###
    # # save sampled images for different estimators
    # # qz_est_name_list = ['GMM_Dirichlet', 'aux_vae']
    # # qz_est_name_list = ['GMM_Dirichlet', 'KDE']
    # # qz_est_name_list = ['GMM_Dirichlet', 'GMM']
    # # qz_est_name_list = ['GMM_1', 'GMM_10', 'GMM_20', 'GMM_100', 'GMM_200']
    # # qz_est_name_list = ['KDE']
    # # qz_est_name_list = ['given_zs']
    # qz_est_name_list = ['aux_vae']
    qz_est_name_list = ['GMM_10']
    qz_samplers = []
    for estimator_name in qz_est_name_list:
        ## Q(z) estimation
        start = time.time()
        if estimator_name.upper().find("AUX_VAE") >= 0:
            second_stage_beta = config.configurations[maj_cfg_idx][minor_cfg_idx]['second_stage_beta']
        else:
            second_stage_beta = 0

        qz_sampler = estimate_density_and_sample.DensityEstimator(training_set=zs_trn,
                                                                  method_name=estimator_name,
                                                                  n_components=n_components,
                                                                  log_dir=log_dir, second_stage_beta=second_stage_beta)
        if estimator_name.upper().find("AUX_VAE") >= 0:
            if os.path.exists(model_path[:-3] + "_" + estimator_name + '_2nd_stage.h5'):
                qz_sampler.fitorload(model_path[:-3] + "_" + estimator_name + '_2nd_stage.h5')
            else:
                qz_sampler.fitorload()
                qz_sampler.save(model_path[:-3] + "_" + estimator_name + '_2nd_stage.h5')
        else:
            qz_sampler.fitorload()
            qz_sampler.save(model_path[:-3] + "_" + estimator_name + '_2nd_stage')
            print ("Time taken to fit " + str(time.time() - start))

        try:
            print("log likelihood for validation set is " + str(qz_sampler.score(valid_embd[0:10000, :])))
            print("log likelihood for train set is " + str(qz_sampler.score(zs_trn[0:10000, :])))
        except NotImplementedError as e:
            print(e)

        start = time.time()
        print("Sampling using " + estimator_name)
        zs = qz_sampler.get_samples(n_samples=10000)
        sampled_images = decoder.predict(zs)
        save_batches_of_images.save_set_of_images(os.path.join(log_dir, estimator_name+'_sampled'), sampled_images)
        print("FID of random samples using " + estimator_name + " is : " +
              str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                                dataset_dir,
                                                os.path.join(log_dir, estimator_name+'_sampled'))))
        qz_samplers.append(qz_sampler)
        print ("Time taken to FID " + str(time.time() - start))

    print("Recon starting")
    np.random.seed(2)
    tf.random.set_random_seed(2)
    # tf.compat.v1.random.set_random_seed(2)
    # Save Validation reconstructions
    recon_images = np.zeros((batches*batch_size,) + sampled_images.shape[1:])
    recon_original = np.zeros((batches*batch_size,) + sampled_images.shape[1:])
    for batch_id in range(batches):
        recon_original[batch_id*batch_size:(batch_id+1)*batch_size] = validation_generator.next()[0]
    print("Going to save original images for recon")
    save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'recon_original'), recon_original)

    for sampler_idx, qz_est_name in enumerate(qz_est_name_list):
        for batch_id in range(batches):
            if qz_est_name.find('aux_vae') >= 0:
                recon_images[batch_id*batch_size:(batch_id+1)*batch_size] = \
                    predict_2stage(encoder, decoder, qz_samplers[sampler_idx],
                                   recon_original[batch_id*batch_size:(batch_id+1)*batch_size])
            else:
                recon_images[batch_id*batch_size:(batch_id+1)*batch_size] = \
                    auto_encoder.predict(recon_original[batch_id*batch_size:(batch_id+1)*batch_size])

        print("Going to save recon images")
        save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'reconstructed_' + qz_est_name), recon_images)
        print("FID of reconstructed samples are : " +
              str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                                dataset_dir,
                                                os.path.join(log_dir, 'reconstructed_' + qz_est_name))))
        l2_validation = np.mean(np.sqrt(np.sum(np.square(recon_images - recon_original),
                                               axis=tuple(range(1, len(recon_original.shape))))))
        print("l2_validation loss: " + str(l2_validation))

    # # Report test error l2
    # recon_images_test = np.zeros((batches * batch_size,) + sampled_images.shape[1:])
    # original_test = np.zeros((batches * batch_size,) + sampled_images.shape[1:])
    # for batch_id in range(batches):
    #     original_test[batch_id * batch_size:(batch_id + 1) * batch_size] = test_generator.next()[0]
    #     recon_images_test[batch_id * batch_size:(batch_id + 1) * batch_size] = \
    #         predict_2stage(encoder, decoder, qz_samplers[sampler_idx],
    #                        original_test[batch_id * batch_size:(batch_id + 1) * batch_size])
    #
    # l2_test = np.mean(np.sqrt(np.sum(np.square(recon_images_test - original_test),
    #                                        axis=tuple(range(1, len(original_test.shape))))))
    # print("l2_test loss: " + str(l2_test))

    # Save interpolation images for FID
    num_interpolation_pts = 3
    z_intrp = np.zeros((10000,) + valid_embd.shape[1:])

    ## linear interpolation
    # pairs = itertools.combinations(pairs_indices, 2)
    for i in range(10000 - 1):
        current_pair = pairs[i]
        z_intrp[i] = (valid_embd[current_pair[0]] + valid_embd[current_pair[1]]) / 2.0

    interpolated_iamges = decoder.predict(z_intrp)
    save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'interpolated_linear_fid'),
                                              interpolated_iamges)
    print("FID of interpolated_linear samples are : " +
          str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                            dataset_dir,
                                            os.path.join(log_dir, 'interpolated_linear_fid'))))

    # linear rescaled interpolation
    z_intrp = np.zeros((10000,) + valid_embd.shape[1:])

    for sampler_idx, qz_est_name in enumerate(qz_est_name_list):
        if qz_est_name.upper().find("AUX_VAE") >= 0:
            test_emb_for_2nd_stage = qz_samplers[sampler_idx].model.encoder.predict(
                (valid_embd-qz_samplers[sampler_idx].model.data_mean)/qz_samplers[sampler_idx].model.data_std)[0]
            for i in range(10000-1):
                current_pair = pairs[i]
                z_intrp[i] = (test_emb_for_2nd_stage[current_pair[0]] + test_emb_for_2nd_stage[current_pair[1]])/2.0
                z_intrp[i] = (z_intrp[i]/np.linalg.norm(z_intrp[i])) * \
                             (np.linalg.norm(test_emb_for_2nd_stage[current_pair[0]]) +
                              np.linalg.norm(test_emb_for_2nd_stage[current_pair[1]]))/2.0
            z_2nd_stage_d_norm = qz_samplers[sampler_idx].model.decoder.predict(z_intrp) * \
                                 qz_samplers[sampler_idx].model.data_std + qz_samplers[sampler_idx].model.data_mean
            interpolated_iamges = decoder.predict(z_2nd_stage_d_norm)
        else:
            for i in range(10000 - 1):
                current_pair = pairs[i]
                z_intrp[i] = (valid_embd[current_pair[0]] + valid_embd[current_pair[1]]) / 2.0
                z_intrp[i] = (z_intrp[i] / np.linalg.norm(z_intrp[i])) * \
                             (np.linalg.norm(valid_embd[current_pair[0]]) + np.linalg.norm(
                                 valid_embd[current_pair[1]])) / 2.0

            interpolated_iamges = decoder.predict(z_intrp)

        save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'interpolated_linear_rescaled_fid_'+
                                                               qz_est_name),
                                                  interpolated_iamges)
        print("FID of interpolated_linear rescaled samples are : " +
              str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                                dataset_dir,
                                                os.path.join(log_dir, 'interpolated_linear_rescaled_fid_' +
                                                             qz_est_name))))

    ## spherical interpolation
    # pairs = itertools.combinations(pairs_indices, 2)
    for i in range(10000-1):
        current_pair = pairs[i]
        z_intrp[i] = interpolations.slerpolate(valid_embd[current_pair[0]], valid_embd[current_pair[1]],
                                               C=None,   # latent_sapce_sampler.get_z_cov(),
                                               num_pts=num_interpolation_pts).T[1]

    interpolated_iamges = decoder.predict(z_intrp)
    save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'interpolated_spherical_fid'), interpolated_iamges)
    print("FID of interpolated_spherical samples are : " +
          str(fid_from_dir_computer.get_fid(config.configurations[maj_cfg_idx][0]['dataset_name'],
                                            dataset_dir,
                                            os.path.join(log_dir, 'interpolated_spherical_fid'))))


    # # save interpolation viz
    num_interpolation_pts = 6
    interpolation_root = os.path.join(log_dir, 'interpolation_viz')
    if not os.path.exists(interpolation_root):
        os.mkdir(interpolation_root)

    for sampler_idx, qz_est_name in enumerate(qz_est_name_list):
        ## linear interpolation
        linear_interpolation_dir = os.path.join(interpolation_root, 'linear_interpolation_viz_' + qz_est_name)
        if not os.path.exists(linear_interpolation_dir):
            os.mkdir(linear_interpolation_dir)
        pairs = pairs_interpolation[config.configurations[maj_cfg_idx][0]['dataset_name']]
        for i in range(20):
            current_pair = pairs[i]
            if qz_est_name.upper().find("AUX_VAE") >= 0:
                zs_test_2nd_stage = qz_samplers[sampler_idx].model.encoder.predict(
                    (zs_test - qz_samplers[sampler_idx].model.data_mean) / qz_samplers[sampler_idx].model.data_std)[
                    0]
                z_intrps_2nd_stage = zs_test_2nd_stage[current_pair[0]].reshape(-1, 1) + \
                                     (zs_test_2nd_stage[current_pair[1]].reshape(-1, 1) -
                                      zs_test_2nd_stage[current_pair[0]].reshape(-1, 1))*\
                                     np.linspace(0, 1, num_interpolation_pts)
                # z_intrps_normalized_2nd_stg = (z_intrps_2nd_stage.T - qz_samplers[sampler_idx].model.data_mean) / \
                #                                qz_samplers[sampler_idx].model.data_std
                z_interps_dnorm = qz_sampler.model.decoder.predict(z_intrps_2nd_stage.T) * \
                                  qz_samplers[sampler_idx].model.data_std + qz_samplers[sampler_idx].model.data_mean
                interpolated_iamges = decoder.predict(z_interps_dnorm)
            else:
                z_intrps = zs_test[current_pair[0]].reshape(-1, 1) + \
                           (zs_test[current_pair[1]].reshape(-1, 1) -
                            zs_test[current_pair[0]].reshape(-1, 1)) * np.linspace(0, 1, num_interpolation_pts)

                interpolated_iamges = decoder.predict(z_intrps.T)

            save_batches_of_images.save_set_of_images(os.path.join(linear_interpolation_dir,
                                                                   str(current_pair[0])+'_'+str(current_pair[1])),
                                                      interpolated_iamges)

    # ## linear interpolation_rescaled
    # for sampler_idx, qz_est_name in enumerate(qz_est_name_list):
    #     linear_interpolation_dir = os.path.join(interpolation_root, 'linear_interpolation_re_scaled_viz' + qz_est_name)
    #     if not os.path.exists(linear_interpolation_dir):
    #         os.mkdir(linear_interpolation_dir)
    #     pairs = pairs_interpolation[config.configurations[maj_cfg_idx][0]['dataset_name']]
    #     for i in range(20):
    #         current_pair = pairs[i]
    #         interp_points = np.linspace(0, 1, num_interpolation_pts)
    #         if qz_est_name.upper().find("AUX_VAE") >= 0:
    #             zs_test_2nd_stage = qz_samplers[sampler_idx].model.encoder.predict(
    #                 (valid_embd - qz_samplers[sampler_idx].model.data_mean) / qz_samplers[sampler_idx].model.data_std)[0]
    #
    #             z_intrps_2nd_stage = zs_test_2nd_stage[current_pair[0]].reshape(-1, 1) + \
    #                                  (zs_test_2nd_stage[current_pair[1]].reshape(-1, 1) -
    #                                  zs_test_2nd_stage[current_pair[0]].reshape(-1, 1)) * interp_points
    #             z_intrps_2nd_stage = (z_intrps_2nd_stage/np.linalg.norm(z_intrps_2nd_stage, axis=0))*\
    #                                  (np.linalg.norm(zs_test[current_pair[0]].reshape(-1, 1), axis=0) +
    #                                  (np.linalg.norm(zs_test[current_pair[1]].reshape(-1, 1), axis=0) -
    #                                   np.linalg.norm(zs_test[current_pair[0]].reshape(-1, 1), axis=0)) * interp_points)
    #             z_interps_dnorm = qz_sampler.model.decoder.predict(z_intrps_2nd_stage.T) * \
    #                               qz_samplers[sampler_idx].model.data_std + qz_samplers[sampler_idx].model.data_mean
    #         else:
    #             pass
    #             # TODO(Partha): Complete it pls
    #         interpolated_iamges = decoder.predict(z_interps_dnorm)
    #         save_batches_of_images.save_set_of_images(os.path.join(linear_interpolation_dir,
    #                                                                str(current_pair[0]) + '_' + str(current_pair[1])),
    #                                                   interpolated_iamges)

    # ## spherical interpolation
    # spherical_interpolation_dir = os.path.join(interpolation_root, 'spherical_interpolation_viz')
    #
    # if not os.path.exists(spherical_interpolation_dir):
    #     os.mkdir(spherical_interpolation_dir)
    # # pairs = itertools.combinations(pairs_indices, 2)
    # for i in range(20):
    #     current_pair = pairs[i]
    #     z_intrps = interpolations.slerpolate(zs_trn[current_pair[0]], zs_trn[current_pair[1]],
    #                                          C=None,  # latent_sapce_sampler.get_z_cov(),
    #                                          num_pts=num_interpolation_pts).T
    #     interpolated_iamges = decoder.predict(z_intrps)
    #     save_batches_of_images.save_set_of_images(os.path.join(spherical_interpolation_dir,
    #                                                            str(current_pair[0]) + '_' + str(current_pair[1])),
    #                                               interpolated_iamges)

    # ## Computing mone carlo approx of log(p(x))
    # # these are training images for finding correct z distribution and not the training images for the original model
    # log_P_X_x = np.zeros(training_images.shape[0])
    # # TODO(Partha): The following must be verified !!
    # sigma_inv_P_X_z = np.eye(training_images[0].flatten().shape[0])  # This must be estiamded for different models i think
    # det_sigma = np.linalg.det(sigma_inv_P_X_z)
    # err_sqr = np.zeros(sampled_images.shape[0])
    # for i, qz_sampler in enumerate(qz_samplers):
    #     zs = qz_sampler.get_samples(training_images.shape[0])
    #     sampled_images = decoder.predict(zs, batch_size=200)
    #     for j, tr_img in enumerate(training_images):
    #         for k, smpl_img in enumerate(sampled_images):
    #             err = tr_img.flatten() - smpl_img.flatten()
    #             err_sqr[k] = -0.5*np.matmul(np.matmul(err.T, sigma_inv_P_X_z), err)
    #         log_P_X_x[j] = misc.logsumexp(err_sqr)
    #         log_P_X_x[j] -= np.log(np.sqrt(2*np.pi*det_sigma))
    #
    #     print("Method: " + qz_est_name_list[i] + ", log(P(X)) = " + str(np.sum(log_P_X_x)) + "var(log(P(X=x))) = " +
    #           str(np.var(log_P_X_x)))


if __name__ == "__main__":
    main()
