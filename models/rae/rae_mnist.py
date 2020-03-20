from keras.regularizers import l2
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Input, BatchNormalization, ReLU, Reshape
from keras.models import Model
from keras.optimizers import Adam
from models.my_layers.spectral_normalized_dense_conv import DenseSN, ConvSN2D, ConvSN2DTranspose
from models.rae import loss_functions


def get_vae_mnist_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                   include_batch_norm, spec_norm_dec_only, recon_loss_func):
    return get_vae_mnist(input_shape, bottleneck_size=8, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=128, spec_norm_dec_only=spec_norm_dec_only,
                         recon_loss_func=recon_loss_func)


def get_vae_mnist_tiny_architecture(input_shape, bottleneck_size, embeding_loss_weight, generator_regs,
                                    generator_reg_types, include_batch_norm, spec_norm_dec_only, recon_loss_func):
    return get_vae_mnist(input_shape, bottleneck_size=bottleneck_size, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=4, spec_norm_dec_only=spec_norm_dec_only,
                         recon_loss_func=recon_loss_func)


def get_vae_mnist(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                  include_batch_norm, num_filter, spec_norm_dec_only, recon_loss_func, verbose=True):
    apply_grad_pen = False
    regularization = None
    _Conv2D = Conv2D
    _Dense = Dense
    _Conv2DTranspose = Conv2DTranspose

    grad_pen_weight = None
    for i, generator_reg_type in enumerate(generator_reg_types):
        if generator_reg_type == 'l2':
            regularization = l2(generator_regs[i])
        elif generator_reg_type == 'grad_pen':
            apply_grad_pen = True
            grad_pen_weight = generator_regs[i]
        elif generator_reg_type == 'spec_norm':
            if not spec_norm_dec_only:
                _Conv2D = ConvSN2D
            _Dense = DenseSN
            _Conv2DTranspose = ConvSN2DTranspose
        elif callable(generator_reg_type):
            regularization = generator_reg_type
        else:
            raise NotImplementedError("Sepecified type of regularization : " + generator_reg_type +
                                      " has not been implemented")

    with tf.name_scope('encoder'):
        e_in = Input(shape=input_shape, name="input_image")

        x = Lambda(lambda x: x*2.0 - 1.0)(e_in)

        x = _Conv2D(num_filter, (4, 4), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = _Conv2D(num_filter*2, (4, 4), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = _Conv2D(num_filter*4, (4, 4), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = _Conv2D(num_filter*8, (4, 4), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)
        z = _Dense(bottleneck_size, activation='linear', name='latent_z')(x)

        encoder = Model(inputs=e_in, outputs=z, name='encoder')

    with tf.name_scope('decoder'):
        d_in = Input(shape=(bottleneck_size,), name='decoder_noise_in')
        x = _Dense(8*8*1024)(d_in)
        x = Reshape((8, 8, 1024))(x)

        x = _Conv2DTranspose(num_filter*4, (4, 4), padding='same', strides=(2, 2), activation='linear',
                            kernel_regularizer=regularization)(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2DTranspose(num_filter*2, (4, 4), padding='same', strides=(2, 2), activation='linear',
                            kernel_regularizer=regularization)(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2DTranspose(input_shape[-1], (4, 4), padding='same', activation='sigmoid',
                            kernel_regularizer=regularization)(x)

        decoder = Model(inputs=d_in, outputs=x, name='decoder')

    with tf.name_scope('full_VAE'):
        loss_func = loss_functions.total_loss(z, beta=embeding_loss_weight, apply_grad_pen=apply_grad_pen,
                                              grad_pen_weight=grad_pen_weight, recon_loss_func=recon_loss_func)
        vae_out = decoder(encoder.outputs[0])
        vae = Model(inputs=e_in, outputs=vae_out, name='vae')
        if apply_grad_pen:
            metrics = [loss_functions.per_pix_recon_loss, loss_functions.embeddig_loss(z),
                       loss_functions.grad_pen_loss(z, None), 'mse']
        else:
            metrics = [loss_functions.per_pix_recon_loss, loss_functions.embeddig_loss(z), 'mse']
        vae.compile(optimizer=Adam(lr=5e-4), loss=loss_func, metrics=metrics)

        if verbose:
            vae.summary()

    return encoder, decoder, vae
