import tensorflow as tf
import keras.backend as K


def total_loss(opts, sample_qz, batch_size, mmd_weight, recon_loss_func=None):
    def _total_loss(y_true, y_pred):
        if recon_loss_func is None:
            recon_loass = per_pix_recon_loss(y_true, y_pred)
        else:
            recon_loass = recon_loss_func(y_true, y_pred)
        mmd_div = mmd_loss(sample_qz, batch_size, opts)()
        return recon_loass + mmd_weight*mmd_div

    return _total_loss


def per_pix_recon_loss(y_true, y_pred):
    # Coppied from https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    # opts['cost'] == 'l2sq'
    # c(x,y) = ||x - y||_2^2
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.05 * tf.reduce_mean(loss)
    return loss


def mmd_loss(sample_qz, batch_size, opts):
    def _mmd_penalty(y_true=0, y_pred=0):
        '''Coppied from https://github.com/tolstikhin/wae/blob/master/wae.py'''
        sample_pz = K.random_normal(shape=(batch_size, opts['zdim']), mean=0.0, stddev=opts['pz_scale'])
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = batch_size
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # if opts['verbose']:
        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
        #         'Maximal Qz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
        #                         'Average Qz squared pairwise distance:')

        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
        #         'Maximal Pz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
        #                         'Average Pz squared pairwise distance:')

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2

        return stat
        # return K.abs(stat)
    return _mmd_penalty