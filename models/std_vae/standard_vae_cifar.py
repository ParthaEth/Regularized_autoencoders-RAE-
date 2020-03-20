from models.std_vae import get_vae_given_enc_dec
from models.rae import rae_cifar, rae_mnist
from keras.layers import Dense
import keras.backend as K


def get_vae_cifar_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                   include_batch_norm, spec_norm_dec_only, recon_loss_func):
    encoder, decoder, _ = rae_cifar.get_vae_cifar_wae_architecture(input_shape,
                                                                   embeding_loss_weight,
                                                                   generator_regs, generator_reg_types,
                                                                   include_batch_norm,
                                                                   spec_norm_dec_only,
                                                                   recon_loss_func=recon_loss_func,
                                                                   verbose=False)

    return build_vae_cifar(encoder, decoder, embeding_loss_weight, recon_loss_func)


def get_vae_cifar(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                  include_batch_norm, num_filter, spec_norm_dec_only, recon_loss_func, constant_sigma):
    encoder, decoder, _ = rae_mnist.get_vae_mnist(input_shape, bottleneck_size=bottleneck_size,
                                                  embeding_loss_weight=embeding_loss_weight,
                                                  generator_regs=generator_regs,
                                                  generator_reg_types=generator_reg_types,
                                                  include_batch_norm=include_batch_norm,
                                                  num_filter=num_filter,
                                                  spec_norm_dec_only=spec_norm_dec_only,
                                                  recon_loss_func=recon_loss_func)
    return build_vae_cifar(encoder, decoder, embeding_loss_weight, recon_loss_func, constant_sigma)


def build_vae_cifar(encoder, decoder, embeding_loss_weight, recon_loss_func, constant_sigma):

    bottleneck_size = K.get_variable_shape(encoder.outputs[0])[-1]
    layer_for_z_sigma = Dense(bottleneck_size, activation='tanh', name='log_sigma')
    return get_vae_given_enc_dec.get_vae(encoder, decoder, embeding_loss_weight, layer_for_z_sigma, recon_loss_func,
                                         constant_sigma)