from models.std_vae import standard_vae_celeba
from models.std_vae import standard_vae_mnist
from models.std_vae import standard_vae_cifar
from models.std_vae import standard_vae_svhn
from my_utility import function_from_name


def get_vae(input_shape, config_dict, major_idx, minor_idx):
    model_name = config_dict[major_idx][minor_idx]['model_name']
    bottleneck_factor = config_dict[major_idx][minor_idx]['bottleneck_factor']
    embeding_loss_weight = config_dict[major_idx][minor_idx]['embedding_weight']
    generator_regs = config_dict[major_idx][minor_idx]['gen_reg_weight']
    reg_types = config_dict[major_idx][minor_idx]['gen_reg_type']
    include_batch_norm = config_dict[major_idx][minor_idx]['include_batch_norm']
    spec_norm_dec_only = config_dict[major_idx][minor_idx]['spec_norm_on_dec_only']
    recon_loss_func = function_from_name.get_recon_loss_func(config_dict[major_idx][minor_idx]['recon_loss_type'])
    num_filters = config_dict[major_idx][minor_idx]['num_filters']
    constant_sigma = config_dict[major_idx][minor_idx]['constant_sigma']

    if model_name.upper() == "MNIST_WAE_PAPER":
        return standard_vae_mnist.get_vae_mnist_wae_architecture(input_shape,
                                                                 embeding_loss_weight,
                                                                 generator_regs=generator_regs,
                                                                 generator_reg_types=reg_types,
                                                                 include_batch_norm=include_batch_norm,
                                                                 spec_norm_dec_only=spec_norm_dec_only,
                                                                 recon_loss_func=recon_loss_func)
    elif model_name.upper() == 'MNIST':
        return standard_vae_mnist.get_vae_mnist(input_shape, bottleneck_size=bottleneck_factor,
                                                embeding_loss_weight=embeding_loss_weight,
                                                generator_regs=generator_regs, generator_reg_types=reg_types,
                                                include_batch_norm=include_batch_norm, num_filter=num_filters,
                                                spec_norm_dec_only=spec_norm_dec_only,
                                                recon_loss_func=recon_loss_func,
                                                constant_sigma=constant_sigma)
    elif model_name.upper() == "CIFAR_WAE_PAPER":
        return standard_vae_cifar.get_vae_cifar_wae_architecture(input_shape,
                                                                 embeding_loss_weight,
                                                                 generator_regs=generator_regs,
                                                                 generator_reg_types=reg_types,
                                                                 include_batch_norm=include_batch_norm,
                                                                 spec_norm_dec_only=spec_norm_dec_only,
                                                                 recon_loss_func=recon_loss_func)
    elif model_name.upper() == 'CIFAR':
        return standard_vae_cifar.get_vae_cifar(input_shape,
                                                embeding_loss_weight=embeding_loss_weight,
                                                generator_regs=generator_regs,
                                                generator_reg_types=reg_types,
                                                include_batch_norm=include_batch_norm,
                                                spec_norm_dec_only=spec_norm_dec_only,
                                                num_filter=num_filters,
                                                bottleneck_size=bottleneck_factor,
                                                recon_loss_func=recon_loss_func,
                                                constant_sigma=constant_sigma)

    elif model_name.upper() == "SVHN_WAE_PAPER":
        return standard_vae_svhn.get_vae_svhn_wae_architecture(input_shape,
                                                               embeding_loss_weight,
                                                               generator_regs=generator_regs,
                                                               generator_reg_types=reg_types,
                                                               include_batch_norm=include_batch_norm,
                                                               spec_norm_dec_only=spec_norm_dec_only,
                                                               recon_loss_func=recon_loss_func)

    elif model_name.upper() == "CELEBA_WAE_PAPER":
        return standard_vae_celeba.get_vae_celeba_wae_architecture(input_shape,
                                                                   embeding_loss_weight,
                                                                   generator_regs=generator_regs,
                                                                   generator_reg_types=reg_types,
                                                                   include_batch_norm=include_batch_norm,
                                                                   spec_norm_dec_only=spec_norm_dec_only,
                                                                   recon_loss_func=recon_loss_func)
    elif model_name.upper() == "CELEBA_WAE_PAPER_MAN_EMB_SZIE":
        return standard_vae_celeba.get_vae_celeba(input_shape,
                                                  embeding_loss_weight=embeding_loss_weight,
                                                  bottleneck_size=bottleneck_factor,
                                                  num_filter=num_filters,
                                                  generator_regs=generator_regs,
                                                  generator_reg_types=reg_types,
                                                  include_batch_norm=include_batch_norm,
                                                  spec_norm_dec_only=spec_norm_dec_only,
                                                  recon_loss_func=recon_loss_func,
                                                  fully_convolutional=False,
                                                  constant_sigma=constant_sigma)
    else:
        raise ValueError("Specified model: " + model_name + " not implemented yet.")