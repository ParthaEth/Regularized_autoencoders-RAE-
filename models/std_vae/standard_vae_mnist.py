from models.std_vae import get_vae_given_enc_dec
from models.rae import rae_mnist
from keras.layers import Dense


def get_vae_mnist_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                   include_batch_norm, spec_norm_dec_only, recon_loss_func):

    return get_vae_mnist(input_shape, bottleneck_size=8, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=128, spec_norm_dec_only=spec_norm_dec_only,
                         recon_loss_func=recon_loss_func, constant_sigma=None)


def get_vae_mnist(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                  include_batch_norm, num_filter, spec_norm_dec_only, recon_loss_func, constant_sigma):
    encoder, decoder, _ = rae_mnist.get_vae_mnist(input_shape, bottleneck_size, embeding_loss_weight,
                                                  generator_regs, generator_reg_types,
                                                  include_batch_norm, num_filter, spec_norm_dec_only,
                                                  recon_loss_func=recon_loss_func, verbose=False)
    layer_for_z_sigma = Dense(bottleneck_size, activation='tanh', name='log_sigma')
    return get_vae_given_enc_dec.get_vae(encoder, decoder, embeding_loss_weight, layer_for_z_sigma, recon_loss_func,
                                         constant_sigma)