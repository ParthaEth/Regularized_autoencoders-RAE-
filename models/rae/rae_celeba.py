from keras.regularizers import l2
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Input, BatchNormalization, ReLU, Reshape
from keras.models import Model
from keras.optimizers import Adam
from models.my_layers.spectral_normalized_dense_conv import DenseSN, ConvSN2D, ConvSN2DTranspose
from models.rae import loss_functions
import keras.backend as K


def get_vae_celeba_wae_architecture(input_shape, embeding_loss_weight, generator_regs, generator_reg_types,
                                    include_batch_norm, spec_norm_dec_only, recon_loss_func):
    return get_vae_cleba(input_shape, bottleneck_size=64, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=128, spec_norm_dec_only=spec_norm_dec_only,
                         fully_convolutional=False, recon_loss_func=recon_loss_func)


def get_celeba_fully_convolutional(input_shape, bottleneck_size, embeding_loss_weight, generator_regs,
                                   generator_reg_types, include_batch_norm, num_filters, spec_norm_dec_only,
                                   recon_loss_func):
    return get_vae_cleba(input_shape, bottleneck_size=bottleneck_size, embeding_loss_weight=embeding_loss_weight,
                         generator_regs=generator_regs, generator_reg_types=generator_reg_types,
                         include_batch_norm=include_batch_norm, num_filter=num_filters,
                         spec_norm_dec_only=spec_norm_dec_only,
                         fully_convolutional=True, recon_loss_func=recon_loss_func)


def get_vae_cleba(input_shape, bottleneck_size, embeding_loss_weight, generator_regs, generator_reg_types,
                  include_batch_norm, num_filter, spec_norm_dec_only, fully_convolutional, recon_loss_func,
                  verbose=True):
    apply_grad_pen = False
    variable_qz_ent = False
    l2_reg = False
    regularization = None
    entropy_qz = None
    _Conv2D = Conv2D
    _Dense = Dense
    _Conv2DTranspose = Conv2DTranspose

    def var_qz_ent_reg(entropy_qz):
        def _var_qz_ent_reg(weight_matrix):
            return K.mean(entropy_qz * K.sum(K.abs(weight_matrix)))
        return _var_qz_ent_reg

    def echo_metric(value_tensor):
        def _echo_value(y_true, y_pred):
            return value_tensor
        return _echo_value

    gard_pen_weight = None
    var_qz_ent_loss_weight = 0
    for i, generator_reg_type in enumerate(generator_reg_types):
        if generator_reg_type == 'l2':
            regularization = l2(generator_regs[i])
        elif generator_reg_type == 'grad_pen':
            apply_grad_pen = True
            gard_pen_weight = generator_regs[i]
        elif generator_reg_type == 'spec_norm':
            if not spec_norm_dec_only:
                _Conv2D = ConvSN2D
            _Dense = DenseSN
            _Conv2DTranspose = ConvSN2DTranspose
        elif generator_reg_type == 'l2_var_qz_ent':
            variable_qz_ent = True
            l2_reg = True
            var_qz_ent_loss_weight = generator_regs[i]
        elif generator_reg_type == 'grad_pen_var_qz_ent':
            variable_qz_ent = True
            apply_grad_pen = True
            gard_pen_weight = generator_regs[i]
            var_qz_ent_loss_weight = generator_regs[i]
        else:
            raise NotImplementedError("Sepecified type of regularization : " + generator_reg_type +
                                      " has not been implemented")
    with tf.name_scope('encoder'):
        # # Build encoder
        e_in = Input(shape=input_shape, name="input_image")

        x = _Conv2D(num_filter, (5, 5), padding='same', activation='linear', strides=(2, 2))(e_in)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2D(num_filter*2, (5, 5), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2D(num_filter*4, (5, 5), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2D(num_filter*8, (5, 5), padding='same', activation='linear', strides=(2, 2))(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU(name='layer_before_z_')(x)

        if fully_convolutional:
            z = _Conv2D(bottleneck_size/4, (5, 5), padding='same', activation='linear', strides=(2, 2),
                        name='latent_z')(x)
        else:
            x = Flatten(name='layer_before_z')(x)
            z = _Dense(bottleneck_size, activation='linear', name='latent_z')(x)

        if variable_qz_ent:
            entropy_qz = _Dense(1, activation='linear', name='entropy_qz')(x)
            # entropy_qz = Lambda(lambda x: 6*(x-0.5))(entropy_qz)
            regularization = None
            encoder = Model(inputs=e_in, outputs=[z, entropy_qz], name='encoder')
        else:
            encoder = Model(inputs=e_in, outputs=z, name='encoder')

    with tf.name_scope('decoder'):
        if fully_convolutional:
            d_in = Input(shape=(input_shape[0]/32, input_shape[1]/32, bottleneck_size/4), name='decoder_noise_in')
            x = _Conv2DTranspose(1024*4, (5, 5), padding='same', activation='linear', strides=(2, 2),
                                 kernel_regularizer=regularization)(d_in)
            x = Lambda(lambda r_x: tf.nn.depth_to_space(r_x, block_size=2))(x)
        else:
            d_in = Input(shape=(bottleneck_size,), name='decoder_noise_in')
            x = _Dense(8*8*1024, activation='linear', kernel_regularizer=regularization)(d_in)
            x = Reshape((8, 8, 1024))(x)

        x = _Conv2DTranspose(num_filter*4, (5, 5), padding='same', strides=(2, 2), activation='linear',
                             kernel_regularizer=regularization)(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2DTranspose(num_filter*2, (5, 5), padding='same', strides=(2, 2), activation='linear',
                             kernel_regularizer=regularization)(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2DTranspose(num_filter, (5, 5), padding='same', strides=(2, 2), activation='linear',
                             kernel_regularizer=regularization)(x)
        if include_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)

        x = _Conv2DTranspose(input_shape[-1], (5, 5), padding='same', activation='sigmoid',
                             kernel_regularizer=regularization)(x)

        decoder = Model(inputs=d_in, outputs=x, name='decoder')

    with tf.name_scope('full_VAE'):
        # z = encoder(e_in)
        vae_out = decoder(encoder.outputs[0])
        metrics = []
        if variable_qz_ent:
            entropy_qz = encoder.outputs[1]
            metrics += [echo_metric(entropy_qz),]
        vae = Model(inputs=e_in, outputs=vae_out, name='vae')
        if apply_grad_pen:
            metrics += [loss_functions.embeddig_loss(z), loss_functions.grad_pen_loss(z, entropy_qz), 'mse']
        else:
            metrics += [loss_functions.embeddig_loss(z), 'mse']

        regularization_loss = None
        if variable_qz_ent and l2_reg:
            regularization_loss = 0
            for trainable_weight_mat in decoder.trainable_weights:
                if trainable_weight_mat.name.find('kernel') >= 0:
                    regularization_loss += var_qz_ent_loss_weight * var_qz_ent_reg(entropy_qz)(trainable_weight_mat)
            metrics += [echo_metric(regularization_loss),]
        vae.compile(optimizer=Adam(lr=1e-3),
                    loss=loss_functions.total_loss(z, beta=embeding_loss_weight, apply_grad_pen=apply_grad_pen,
                                                   grad_pen_weight=gard_pen_weight, recon_loss_func=recon_loss_func,
                                                   entropy_qz=entropy_qz, regularization_loss=regularization_loss),
                    metrics=metrics)

        if verbose:
            vae.summary()

    return encoder, decoder, vae