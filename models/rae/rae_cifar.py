from models.rae import rae_mnist


def get_vae_cifar_wae_architecture(input_shape, embeding_loss_weight, generator_reg, generator_reg_type,
                                   include_batch_norm, spec_norm_dec_only, recon_loss_func, verbose=True):
    return rae_mnist.get_vae_mnist(input_shape, bottleneck_size=64,
                                   embeding_loss_weight=embeding_loss_weight,
                                   generator_reg=generator_reg, generator_reg_type=generator_reg_type,
                                   include_batch_norm=include_batch_norm, num_filter=128,
                                   spec_norm_dec_only=spec_norm_dec_only, recon_loss_func=recon_loss_func,
                                   verbose=verbose)


def get_vae_cifar_tiny_architecture(input_shape, bottleneck_size, embeding_loss_weight, generator_reg,
                                    generator_reg_type, include_batch_norm, spec_norm_dec_only, recon_loss_func):
    return rae_mnist.get_vae_mnist(input_shape, bottleneck_size=bottleneck_size,
                                   embeding_loss_weight=embeding_loss_weight,
                                   generator_reg=generator_reg, generator_reg_type=generator_reg_type,
                                   include_batch_norm=include_batch_norm, num_filter=4,
                                   spec_norm_dec_only=spec_norm_dec_only, recon_loss_func=recon_loss_func)