from models.wae_mmd import get_wae_given_enc_dec
from models.rae import rae_mnist


def get_wae_mnist_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                   include_batch_norm, spec_norm_dec_only, batch_size, recon_loss_func):

    return get_wae_mnist(input_shape, bottleneck_size=8, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=128,
                         spec_norm_dec_only=spec_norm_dec_only, fully_convolutional=False,
                         batch_size=batch_size, recon_loss_func=recon_loss_func)


def get_wae_mnist(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                  include_batch_norm, num_filter, spec_norm_dec_only, fully_convolutional, batch_size,
                  recon_loss_func):

    if fully_convolutional:
        raise NotImplementedError('Fully convolutional architecture for MNIST has not been implemented')

    encoder, decoder, _ = rae_mnist.get_vae_mnist(input_shape, bottleneck_size, embeding_loss_weight,
                                                  generator_regs, generator_reg_types,
                                                  include_batch_norm, num_filter, spec_norm_dec_only,
                                                  recon_loss_func=recon_loss_func,
                                                  verbose=False)
    return get_wae_given_enc_dec.get_wae(encoder, decoder, embeding_loss_weight, batch_size, recon_loss_func)