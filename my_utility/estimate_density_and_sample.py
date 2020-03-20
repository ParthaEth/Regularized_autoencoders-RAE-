from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
import keras
from keras.layers import Dense, Input
from keras.models import Model
import tensorflow as tf
from models.std_vae import get_vae_given_enc_dec
from keras.losses import mean_squared_error
import numpy as np
import pickle
import os


class DensityEstimator:
    def __init__(self, training_set, method_name, n_components=None, log_dir=None, second_stage_beta=None):
        self.log_dir = log_dir
        self.training_set = training_set
        self.fitting_done = False
        self.method_name = method_name
        self.second_density_mdl = None
        self.skip_fitting_and_sampling = False
        if method_name == "GMM_Dirichlet":
            self.model = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full',
                                                         weight_concentration_prior=1.0/n_components)
        elif method_name == "GMM":
            self.model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_1":
            self.model = mixture.GaussianMixture(n_components=1, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_10":
            self.model = mixture.GaussianMixture(n_components=10, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_20":
            self.model = mixture.GaussianMixture(n_components=20, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_100":
            self.model = mixture.GaussianMixture(n_components=100, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)
        elif method_name == "GMM_200":
            self.model = mixture.GaussianMixture(n_components=200, covariance_type='full', max_iter=2000,
                                                 verbose=2, tol=1e-3)

        elif method_name.find("aux_vae") >= 0:
            have_2nd_density_est = False
            if method_name[8:] != "":
                self.second_density_mdl = method_name[8:]
                have_2nd_density_est = True
            self.model = VaeModelWrapper(input_shape=(training_set.shape[-1], ),
                                         latent_space_dim=training_set.shape[-1],
                                         have_2nd_density_est=have_2nd_density_est,
                                         log_dir=self.log_dir, sec_stg_beta=second_stage_beta)

        elif method_name == "given_zs":
            files = os.listdir(log_dir)
            for z_smpls in files:
                if z_smpls.endswith('.npy'):
                    break
            self.z_smps = np.load(os.path.join(log_dir, z_smpls))
            self.skip_fitting_and_sampling = True

        elif method_name.upper() == "KDE":
            self.model = KernelDensity(kernel='gaussian', bandwidth=0.425)
            # self.model = KernelDensity(kernel='tophat', bandwidth=15)
        else:
            raise NotImplementedError("Method specified : " + str(method_name) + " doesn't have an implementation yet.")

    def fitorload(self, file_name=None):
        if not self.skip_fitting_and_sampling:
            if file_name is None:
                self.model.fit(self.training_set, self.second_density_mdl)
            else:
                self.model.load(file_name)

        self.fitting_done = True

    def score(self, X, y=None):
        if self.method_name.upper().find("AUX_VAE") >= 0 or self.skip_fitting_and_sampling:
            raise NotImplementedError("Log likelihood evaluation for VAE is difficult. or skipped")
        else:
            return self.model.score(X, y)

    def save(self, file_name):
        if not self.skip_fitting_and_sampling:
            if self.method_name.find('vae') >= 0:
                self.model.save(file_name)
            else:
                with open(file_name, 'wb') as f:
                    pickle.dump(self.model, f)


    def reconstruct(self, input_batch):
        if self.method_name.upper().find("AUX_VAE") < 0:
            raise ValueError("Non autoencoder style density estimator: " + self.method_name)
        return self.model.reconstruct(input_batch)

    def get_samples(self, n_samples):
        if not self.skip_fitting_and_sampling:
            if not self.fitting_done:
                self.fitorload()
            scrmb_idx = np.array(range(n_samples))
            np.random.shuffle(scrmb_idx)
            if self.log_dir is not None:
                pickle_path = os.path.join(self.log_dir, self.method_name+'_mdl.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.model, f)
            if self.method_name.upper() == "GMM_DIRICHLET" or self.method_name.upper() == "AUX_VAE" \
                    or self.method_name.upper() == "GMM" or self.method_name.upper() == "GMM_1" \
                    or self.method_name.upper() == "GMM_10" or self.method_name.upper() == "GMM_20" \
                    or self.method_name.upper() == "GMM_100" or self.method_name.upper() == "GMM_200"\
                    or self.method_name.upper().find("AUX_VAE") >= 0:
                return self.model.sample(n_samples)[0][scrmb_idx, :]
            else:
                return np.random.shuffle(self.model.sample(n_samples))[scrmb_idx, :]
        else:
            return self.z_smps


class VaeModelWrapper:
    def __init__(self, input_shape, latent_space_dim, have_2nd_density_est, log_dir, sec_stg_beta):
        self.log_dir = log_dir
        self.latent_space_dim = latent_space_dim
        self.have_2nd_density_est = have_2nd_density_est
        self.sec_stg_beta = sec_stg_beta
        with tf.name_scope('Encoder'):
            e_in = Input(shape=input_shape)
            x = Dense(1024, activation='relu')(e_in)
            x = Dense(1024, activation='relu')(x)
            z = Dense(latent_space_dim, activation='linear')(x)
            encoder = Model(inputs=e_in, outputs=z)

            layer_for_z_sigma = Dense(latent_space_dim, activation='tanh')

        with tf.name_scope('Decoder'):
            d_in = Input(shape=(latent_space_dim, ))
            x = Dense(1024, activation='relu')(d_in)
            x = Dense(1024, activation='relu')(x)
            d_out = Dense(input_shape[0], activation='linear')(x)
            decoder = Model(inputs=d_in, outputs=d_out)

        self.encoder, self.decoder, self.auto_encoder = get_vae_given_enc_dec.get_vae(
            encoder, decoder,
            embeding_loss_weight=self.sec_stg_beta,
            layer_for_z_sigma=layer_for_z_sigma,
            recon_loss_func=mean_squared_error,
            constant_sigma=None)

    def load(self, file_name):
        self.auto_encoder.load_weights(file_name)
        normalization = np.load(file_name+"_normalization.npz")
        self.data_std = normalization["data_std"]
        self.data_mean = normalization["data_mean"]
        if self.have_2nd_density_est:
            with open(file_name + '_2nd_density_est', 'rb') as f:
                self.u_est = pickle.load(f)

    def reconstruct(self, input_batch):
        inp_batch_normalized = (input_batch[:] - self.data_mean)/self.data_std
        inp_batch_recon = self.auto_encoder.predict(inp_batch_normalized)
        inp_batch_recon_dnorm = inp_batch_recon*self.data_std + self.data_mean
        return inp_batch_recon_dnorm

    def save(self, file_name):
        self.auto_encoder.save_weights(file_name)
        np.savez(file_name+"_normalization.npz", data_std=self.data_std, data_mean=self.data_mean)
        if self.have_2nd_density_est:
            with open(file_name+'_2nd_density_est', 'wb') as f:
                pickle.dump(self.u_est, f)

    def fit(self, training_data, de_2nd_name=None):
        self.data_mean = np.mean(training_data, axis=0)
        self.data_std = np.std(training_data, axis=0)
        print('data_mean + ' + str(self.data_mean))
        print('data_std + ' + str(self.data_std))

        self.training_data_normalized = (training_data[:] - self.data_mean)/self.data_std

        # callbacks
        cbs = []
        reduce_on_pl_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                                                            mode='auto',  min_delta=0.0001, cooldown=0, min_lr=0)
        cbs.append(reduce_on_pl_cb)

        self.auto_encoder.fit(self.training_data_normalized, self.training_data_normalized, batch_size=64, epochs=100,
                              validation_split=0.1, verbose=1, callbacks=cbs)

        if self.have_2nd_density_est:
            u_train = self.encoder.predict(self.training_data_normalized)[0]
            self.u_est = DensityEstimator(training_set=u_train, method_name=de_2nd_name, n_components=None,
                                          log_dir=self.log_dir)
            self.u_est.fitorload()

    def sample(self, n_samples):
        if self.have_2nd_density_est:
            u_s = self.u_est.model.sample(n_samples)[0]
        else:
            u_s = np.random.normal(0, 1, size=(n_samples, self.latent_space_dim))

        z_s = self.decoder.predict(u_s, batch_size=200)
        z_s = z_s*self.data_std + self.data_mean
        return z_s, None