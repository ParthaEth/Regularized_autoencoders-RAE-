from models.wae_mmd import wae_celeba
from models.wae_mmd import wae_mnist
# import wae_cifar
# import wae_svhn
from my_utility import function_from_name


def get_wae(input_shape, config_dict, major_idx, minor_idx):
    model_name = config_dict[major_idx][minor_idx]['model_name']
    bottleneck_factor = config_dict[major_idx][minor_idx]['bottleneck_factor']
    embeding_loss_weight = config_dict[major_idx][minor_idx]['embedding_weight']
    generator_regs = config_dict[major_idx][minor_idx]['gen_reg_weight']
    reg_types = config_dict[major_idx][minor_idx]['gen_reg_type']
    include_batch_norm = config_dict[major_idx][minor_idx]['include_batch_norm']
    spec_norm_dec_only = config_dict[major_idx][minor_idx]['spec_norm_on_dec_only']
    batch_size = config_dict[major_idx][0]['batch_size']
    recon_loss_func = function_from_name.get_recon_loss_func(config_dict[major_idx][minor_idx]['recon_loss_type'])
    num_filters = config_dict[major_idx][minor_idx]['num_filters']

    if model_name.upper() == "MNIST_WAE_PAPER":
        return wae_mnist.get_wae_mnist_wae_architecture(input_shape,
                                                        embeding_loss_weight,
                                                        generator_regs=generator_regs,
                                                        generator_reg_types=reg_types,
                                                        include_batch_norm=include_batch_norm,
                                                        spec_norm_dec_only=spec_norm_dec_only,
                                                        batch_size=batch_size, recon_loss_func=recon_loss_func)
    elif model_name.upper() == "MNIST":
        return wae_mnist.get_wae_mnist(input_shape, bottleneck_size=bottleneck_factor,
                                       embeding_loss_weight=embeding_loss_weight,
                                       generator_regs=generator_regs, generator_reg_types=reg_types,
                                       include_batch_norm=include_batch_norm, num_filter=num_filters,
                                       spec_norm_dec_only=spec_norm_dec_only, fully_convolutional=False,
                                       batch_size=batch_size, recon_loss_func=recon_loss_func)

    elif model_name.upper() == "CIFAR_WAE_PAPER":
        raise NotImplementedError("Specified model: " + model_name + " not implemented yet.")
        return wae_cifar.get_wae_cifar_wae_architecture(input_shape,
                                                                 embeding_loss_weight,
                                                                 generator_regs=generator_regs,
                                                                 generator_reg_types=reg_types,
                                                                 include_batch_norm=include_batch_norm,
                                                                 spec_norm_dec_only=spec_norm_dec_only)
    elif model_name.upper() == "CIFAR":
        # MNIST model usage is intentional only difference used is the size of the bottleneck size
        return wae_mnist.get_wae_mnist(input_shape, bottleneck_size=bottleneck_factor,
                                       embeding_loss_weight=embeding_loss_weight,
                                       generator_regs=generator_regs, generator_reg_types=reg_types,
                                       include_batch_norm=include_batch_norm, num_filter=num_filters,
                                       spec_norm_dec_only=spec_norm_dec_only, fully_convolutional=False,
                                       batch_size=batch_size, recon_loss_func=recon_loss_func)
    elif model_name.upper() == "SVHN_WAE_PAPER":
        raise NotImplementedError("Specified model: " + model_name + " not implemented yet.")
        return wae_svhn.get_wae_svhn_wae_architecture(input_shape,
                                                               embeding_loss_weight,
                                                               generator_regs=generator_regs,
                                                               generator_reg_types=reg_types,
                                                               include_batch_norm=include_batch_norm,
                                                               spec_norm_dec_only=spec_norm_dec_only)

    elif model_name.upper() == "CELEBA_WAE_PAPER":
        return wae_celeba.get_wae_celeba_wae_architecture(input_shape,
                                                          embeding_loss_weight,
                                                          generator_regs=generator_regs,
                                                          generator_reg_types=reg_types,
                                                          include_batch_norm=include_batch_norm,
                                                          spec_norm_dec_only=spec_norm_dec_only,
                                                          batch_size=batch_size,
                                                          recon_loss_func=recon_loss_func)

    elif model_name.upper() == "CELEBA_WAE_PAPER_MAN_EMB_SZIE":
        return wae_celeba.get_wae_celeba(input_shape, bottleneck_size=bottleneck_factor,
                                         embeding_loss_weight=embeding_loss_weight,
                                         generator_regs=generator_regs, generator_reg_types=reg_types,
                                         include_batch_norm=include_batch_norm, num_filter=num_filters,
                                         spec_norm_dec_only=spec_norm_dec_only, fully_convolutional=False,
                                         batch_size=batch_size, recon_loss_func=recon_loss_func)
    else:
        raise NotImplementedError("Specified model: " + model_name + " not implemented yet.")