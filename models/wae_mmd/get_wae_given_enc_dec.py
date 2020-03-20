from keras.layers import Lambda
import tensorflow as tf
from keras.models import Model
from models.wae_mmd import loss_functions
from keras.optimizers import Adam
import keras.backend as K


def get_wae(encoder, decoder, embeding_loss_weight, batch_size, recon_loss_func):

    with tf.name_scope('full_WAE'):
        bottleneck_size = K.get_variable_shape(encoder.outputs[0])[-1]
        opts = {'mmd_kernel': 'IMQ', 'pz_scale': 1.0, 'pz': 'normal', 'zdim': bottleneck_size}
        e_in = encoder.inputs
        if bottleneck_size == 64: # Dirty hack as WAE for CELEBA is trained usng noisy input as suggested in WAE paper others are not Make it uniform
            e_in_noisy = Lambda(lambda x: x + K.clip(K.random_normal(K.shape(x),
                                                                     mean=0.0, stddev=0.01), -0.01, 0.01))(e_in[0])
            q_zs = encoder(e_in_noisy)
        else:
            q_zs = encoder.outputs[0]
        vae_out = decoder(q_zs)
        vae = Model(inputs=e_in, outputs=vae_out, name='vae')
        vae.compile(optimizer=Adam(lr=1e-3),
                    loss=loss_functions.total_loss(opts, q_zs, batch_size, embeding_loss_weight, recon_loss_func),
                    metrics=[loss_functions.mmd_loss(q_zs, batch_size, opts), 'mse'])
        vae.summary()

    return encoder, decoder, vae
