import keras.backend as K
from keras.losses import mse


def get_sampler(latent_dim):
    def sample_z(args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], latent_dim), mean=0., stddev=1.)
        return mu + K.exp(log_sigma) * eps
    return sample_z


def loss_kl_divergence(mu, log_sigma):
    def kl_loss(y_true, y_pred):
        # Don't panic about the - sign. It has been pushed though into the barcket
        kl = 0.5 * K.sum(K.exp(2*log_sigma) + K.square(mu) - 1. - 2*log_sigma, axis=1)
        return kl

    return kl_loss


def total_loss(mu, log_sigma, kl_weight=1, recon_loss_func=None):
    def _vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for eatch data in minibatch """
        # E[log P(X|z)]
        if recon_loss_func is None:
            recon = mse(y_pred, y_true)
        else:
            recon = recon_loss_func(y_true, y_pred)

        # D_KL(Q(z|X) || P(z|X)); calculate in closed from as both dist. are Gaussian
        kl = loss_kl_divergence(mu, log_sigma)(0, 0)

        return K.sum(recon, axis=list(range(1, recon.shape.ndims))) + kl_weight * kl

    return _vae_loss