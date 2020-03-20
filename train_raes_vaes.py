import sys
import errno
import keras
from my_utility import my_callbacks
import os
from models.rae import make_raes
from models.std_vae import std_vae
from models.wae_mmd import wae_mmd
from dataloaders.dataloader import DataLoader
from configurations import config
from my_utility import config_parser
from my_utility.accumulate_batches_of_data_frm_generator import get_n_batches_of_input
from my_utility import save_restore_model_state
from my_utility.my_callbacks import LatentSpaceSampler
from my_utility import save_batches_of_images


def main():
    starting_epoch = 0
    test_mode = False
    resume_training = False
    if len(sys.argv) > 2:
        if sys.argv[2].upper() == 'TEST':
            test_mode = True
        else:
            raise ValueError("Can only deal with only 'TEST' as second argument. While first is config number but "
                             "provided " + str(sys.argv))

    maj_cfg_idx, minor_cfg_idx = config_parser.get_config_idxs(int(sys.argv[1]), config.configurations)
    log_root = config.configurations[maj_cfg_idx][0]['log_root']
    log_root = os.path.join(log_root, str(maj_cfg_idx))
    try:
        os.makedirs(log_root)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    log_dir = os.path.join(log_root, config.configurations[maj_cfg_idx][minor_cfg_idx]['expt_name'] +\
                                     '_' + str(minor_cfg_idx))
    if not test_mode:
        tb_log_dir = os.path.join(log_dir, 'tb')
        if os.path.exists(log_dir):
            resume_training = True
        else:
            os.mkdir(log_dir)
            os.mkdir(tb_log_dir)

    model_name = config.configurations[maj_cfg_idx][0]['base_model_name'] + '_' + \
                 config.configurations[maj_cfg_idx][0]['dataset_name'] + '.h5'
    model_path = os.path.join(log_dir, model_name)

    (train_generator, validation_generator, _), input_shape, (train_steps, validation_steps, _) = \
        DataLoader(batch_size=config.configurations[maj_cfg_idx][0]['batch_size']).\
            get_data_loader(dataset_name=config.configurations[maj_cfg_idx][0]['dataset_name'], shuffle=True)

    if config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('RAE') >= 0:
        encoder, decoder, auto_encoder = make_raes.get_vae(input_shape, config.configurations, maj_cfg_idx,
                                                           minor_cfg_idx)
    elif config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('WAE') >= 0:
        encoder, decoder, auto_encoder = wae_mmd.get_wae(input_shape, config.configurations, maj_cfg_idx,
                                                         minor_cfg_idx)
    elif config.configurations[maj_cfg_idx][0]['base_model_name'].upper().find('STD_VAE') >= 0:
        encoder, decoder, auto_encoder = std_vae.get_vae(input_shape, config.configurations, maj_cfg_idx,
                                                         minor_cfg_idx)
    else:
        raise NotImplementedError("No implemntation for " +
                                  str(config.configurations[maj_cfg_idx][0]['base_model_name']) + " found.")

    if resume_training:
        starting_epoch = save_restore_model_state.restore_model_state(
            auto_encoder, checkpoint_path=model_path)

    if not test_mode:
        callbacks = []
        # embeddings_data = get_n_batches_of_input(10, validation_generator)
        tb_call_back = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=False) #,
                                                   # embeddings_freq=1,
                                                   # embeddings_layer_names=['latent_z',],
                                                   # embeddings_data=embeddings_data)
        # tb_call_back = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=False)
        callbacks.append(tb_call_back)
        log_fid = False
        num_fid_samples = config.configurations[maj_cfg_idx][0]['log_fid_with_smpls']
        if num_fid_samples is not None and num_fid_samples != 0:
            log_fid = True
        recon_image_logger = my_callbacks.SaveReconstructedImages(
            epoch_freq = 5,
            models=(encoder, decoder, auto_encoder),
            test_subset=validation_generator,
            log_dir=tb_log_dir, num_samples=num_fid_samples,
            get_writer_frm=tb_call_back,
            log_fid=log_fid,
            last_epoch=config.configurations[maj_cfg_idx][0]['epochs']-1,
            num_last_epoch_fid_samples=config.configurations[maj_cfg_idx][0]['num_last_epoch_fid_samples'])
        callbacks.append(recon_image_logger)

        red_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        callbacks.append(red_on_plateau)

        mdl_check_point = keras.callbacks.ModelCheckpoint(model_path+'_best', monitor='val_loss',
                                                          save_best_only=True, save_weights_only=True, mode='auto',
                                                          period=1)
        callbacks.append(mdl_check_point)

        try:
            # auto_encoder.load_weights(prev_dir_path + '/' + model_name)
            auto_encoder.fit_generator(generator=train_generator,
                                       steps_per_epoch=train_steps,
                                       epochs=config.configurations[maj_cfg_idx][0]['epochs'],
                                       callbacks=callbacks,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps,
                                       workers=1,
                                       use_multiprocessing=False,
                                       initial_epoch=starting_epoch)
        except ValueError as e:
            raise ValueError("Most likely because you forgot to specify latent layer name as 'latent_z' {0}".format(e))
        finally:
            save_restore_model_state.save_model_state(auto_encoder, model_path, recon_image_logger.get_current_epoch())
            print("<<<<<<<<<<<<<<< Model : " + model_name + " saved >>>>>>>>>>>>>>>>>>>>>>>")
    else:
        batches = 100
        # auto_encoder.load_weights(model_path + '_model_weights.h5')
        auto_encoder.load_weights(model_path)
        latent_sapce_sampler = LatentSpaceSampler(encoder)
        valid_images = get_n_batches_of_input(batches, validation_generator)
        zs = latent_sapce_sampler.get_zs(valid_images)
        sampled_images = decoder.predict(zs)
        save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'sampled'), sampled_images)

        recon_images = auto_encoder.predict_generator(validation_generator, steps=batches)
        save_batches_of_images.save_set_of_images(os.path.join(log_dir, 'reconstructed'), recon_images)


if __name__ == "__main__":
    main()