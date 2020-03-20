from keras.layers import Lambda
import tensorflow as tf
from keras.models import Model
from models.std_vae import loss_functions
from keras.optimizers import Adam
import keras.backend as K


def get_vae(encoder, decoder, embeding_loss_weight, layer_for_z_sigma, recon_loss_func, constant_sigma):

    last_but_one_encoder_layer_output = encoder.get_layer(index=-2).output
    with tf.name_scope('encoder'):
        # log_sigma = _Dense(bottleneck_size, activation='tanh')(last_but_one_encoder_layer_output)
        e_in = encoder.inputs
        if constant_sigma is None:
            log_sigma = layer_for_z_sigma(last_but_one_encoder_layer_output)
            log_sigma = Lambda(lambda x: 5 * x, name='z_sigma')(log_sigma)
            encoder = Model(inputs=e_in, outputs=encoder.outputs + [log_sigma], name='std_vae_encoder_model')
        else:
            # Very nasty hack. Takes an input but always returns the same constant value!
            log_sigma = Lambda(lambda x: K.log(constant_sigma))(last_but_one_encoder_layer_output)

    with tf.name_scope('full_VAE'):
        mu = encoder.outputs[0]
        bottleneck_size = K.get_variable_shape(encoder.outputs[0])[-1]
        z = Lambda(loss_functions.get_sampler(bottleneck_size))([mu, log_sigma])
        vae_out = decoder(z)
        vae = Model(inputs=e_in, outputs=vae_out, name='vae')
        vae.compile(optimizer=Adam(lr=1e-4), loss=loss_functions.total_loss(mu, log_sigma,
                                                                            kl_weight=embeding_loss_weight,
                                                                            recon_loss_func=recon_loss_func),
                    metrics=[loss_functions.loss_kl_divergence(mu, log_sigma), 'mse'])
        vae.summary()

    return encoder, decoder, vae
