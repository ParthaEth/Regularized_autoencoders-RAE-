from models.rae import rae_celeba
from models.rae import rae_mnist
from models.rae import rae_cifar
from models.rae import rae_svhn
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

    if model_name.upper() == "MNIST_WAE_PAPER":
        return rae_mnist.get_vae_mnist_wae_architecture(input_shape,
                                                        embeding_loss_weight,
                                                        generator_regs=generator_regs,
                                                        generator_reg_types=reg_types,
                                                        include_batch_norm=include_batch_norm,
                                                        spec_norm_dec_only=spec_norm_dec_only,
                                                        recon_loss_func=recon_loss_func)
    if model_name.upper() == "MNIST_TINY":
        return rae_mnist.get_vae_mnist_tiny_architecture(input_shape,
                                                         bottleneck_factor,
                                                         embeding_loss_weight=embeding_loss_weight,
                                                         generator_regs=generator_regs,
                                                         generator_reg_types=reg_types,
                                                         include_batch_norm=include_batch_norm,
                                                         spec_norm_dec_only=spec_norm_dec_only,
                                                         recon_loss_func=recon_loss_func)
    elif model_name.upper() == "MNIST":
        return rae_mnist.get_vae_mnist(input_shape, bottleneck_size=bottleneck_factor,
                                       embeding_loss_weight=embeding_loss_weight,
                                       generator_regs=generator_regs, generator_reg_types=reg_types,
                                       include_batch_norm=include_batch_norm, num_filter=num_filters,
                                       spec_norm_dec_only=spec_norm_dec_only,
                                       recon_loss_func=recon_loss_func)

    elif model_name.upper() == "CELEBA_WAE_PAPER":
        return rae_celeba.get_vae_celeba_wae_architecture(input_shape,
                                                          embeding_loss_weight,
                                                          generator_regs=generator_regs,
                                                          generator_reg_types=reg_types,
                                                          include_batch_norm=include_batch_norm,
                                                          spec_norm_dec_only=spec_norm_dec_only,
                                                          recon_loss_func=recon_loss_func)
    elif model_name.upper() == "CELEBA_WAE_PAPER_MAN_EMB_SZIE":
        return rae_celeba.get_vae_cleba(input_shape,
                                        num_filter=num_filters,
                                        embeding_loss_weight=embeding_loss_weight,
                                        bottleneck_size=bottleneck_factor,
                                        generator_regs=generator_regs,
                                        generator_reg_types=reg_types,
                                        include_batch_norm=include_batch_norm,
                                        spec_norm_dec_only=spec_norm_dec_only,
                                        recon_loss_func=recon_loss_func,
                                        fully_convolutional=False)

    elif model_name.upper() == "CELEBA_FULLY_CONVOLUTIONAL":
        return rae_celeba.get_celeba_fully_convolutional(input_shape,
                                                         bottleneck_factor,
                                                         embeding_loss_weight,
                                                         num_filters=num_filters,
                                                         generator_regs=generator_regs,
                                                         generator_reg_types=reg_types,
                                                         include_batch_norm=include_batch_norm,
                                                         spec_norm_dec_only=spec_norm_dec_only,
                                                         recon_loss_func=recon_loss_func)

    elif model_name.upper() == "CIFAR_WAE_PAPER":
        return rae_cifar.get_vae_cifar_wae_architecture(input_shape,
                                                        embeding_loss_weight,
                                                        generator_regs=generator_regs,
                                                        generator_reg_types=reg_types,
                                                        include_batch_norm=include_batch_norm,
                                                        spec_norm_dec_only=spec_norm_dec_only,
                                                        recon_loss_func=recon_loss_func)
    elif model_name.upper() == "CIFAR_TINY":
        return rae_cifar.get_vae_cifar_tiny_architecture(input_shape,
                                                         bottleneck_factor,
                                                         embeding_loss_weight=embeding_loss_weight,
                                                         generator_regs=generator_regs,
                                                         generator_reg_types=reg_types,
                                                         include_batch_norm=include_batch_norm,
                                                         spec_norm_dec_only=spec_norm_dec_only,
                                                         recon_loss_func=recon_loss_func)
    elif model_name.upper() == "CIFAR":
        # Mnist model there is intentional. Only factor that is different is the bottleneck size
        return rae_mnist.get_vae_mnist(input_shape, bottleneck_size=bottleneck_factor,
                                       embeding_loss_weight=embeding_loss_weight,
                                       generator_regs=generator_regs, generator_reg_types=reg_types,
                                       include_batch_norm=include_batch_norm, num_filter=num_filters,
                                       spec_norm_dec_only=spec_norm_dec_only,
                                       recon_loss_func=recon_loss_func)

    elif model_name.upper() == "SVHN_WAE_PAPER":
        return rae_svhn.get_vae_svhn_wae_architecture(input_shape,
                                                      embeding_loss_weight,
                                                      generator_regs=generator_regs,
                                                      generator_reg_types=reg_types,
                                                      include_batch_norm=include_batch_norm,
                                                      spec_norm_dec_only=spec_norm_dec_only,
                                                      recon_loss_func=recon_loss_func)
    else:
        raise ValueError("Specified model: " + model_name + " not implemented yet.")