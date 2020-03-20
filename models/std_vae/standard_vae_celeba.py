from models.std_vae import get_vae_given_enc_dec
from models.rae import rae_celeba
from keras.layers import Dense


def get_vae_celeba_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                   include_batch_norm, spec_norm_dec_only, recon_loss_func):

    return get_vae_celeba(input_shape, bottleneck_size=64, embeding_loss_weight=embeding_loss_weight,
                          generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                          include_batch_norm=include_batch_norm, num_filter=128,
                          spec_norm_dec_only=spec_norm_dec_only, fully_convolutional=False,
                          recon_loss_func=recon_loss_func)


def get_vae_celeba(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                   include_batch_norm, num_filter, spec_norm_dec_only, fully_convolutional, recon_loss_func,
                   constant_sigma):
    encoder, decoder, _ = rae_celeba.get_vae_cleba(input_shape, bottleneck_size, embeding_loss_weight,
                                                   generator_regs, generator_reg_types,
                                                   include_batch_norm, num_filter, spec_norm_dec_only,
                                                   fully_convolutional, recon_loss_func=recon_loss_func,
                                                   verbose=False)
    layer_for_z_sigma = Dense(bottleneck_size, activation='tanh', name='log_sigma')
    return get_vae_given_enc_dec.get_vae(encoder, decoder, embeding_loss_weight, layer_for_z_sigma, recon_loss_func,
                                         constant_sigma)