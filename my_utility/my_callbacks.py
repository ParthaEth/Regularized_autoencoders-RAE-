import keras
import types
import tensorflow as tf
import keras.backend as K
import numpy as np
from tensorflow.core.framework import graph_pb2
import tarfile


class FIDComputer:

    def __init__(self, max_smpls_in_batch):
        self.fid = None
        self.max_smpls_in_a_batch = max_smpls_in_batch

    def inception_based_classifier(self, images):
        """Expect images in the scale of 0-255"""
        return tf.contrib.gan.eval.run_inception(tf.contrib.gan.eval.preprocess_image(images),
                                                 output_tensor='pool_3:0')

    def get_fid(self, sess, real_images, generated_images):
        if real_images.shape[-1] == 1:
            real_images_colored = np.repeat(real_images, repeats=3, axis=-1)
            generated_images_colored = np.repeat(generated_images, repeats=3, axis=-1)
        else:
            real_images_colored = real_images
            generated_images_colored = generated_images
        if self.fid is None:
            num_batches = real_images_colored.shape[0] / self.max_smpls_in_a_batch
            image_shape = (None,) + real_images_colored.shape[1:]
            self.real_images = tf.placeholder(tf.float32, shape=image_shape)
            self.generated_images = tf.placeholder(tf.float32, shape=image_shape)
            self.fid = tf.contrib.gan.eval.frechet_classifier_distance(tf.multiply(self.real_images, 255.0),
                                                                       tf.multiply(self.generated_images, 255.0),
                                                                       classifier_fn=self.inception_based_classifier,
                                                                       num_batches=num_batches)
        fid = sess.run(self.fid, feed_dict={self.real_images: real_images_colored,
                                            self.generated_images: generated_images_colored})
        return fid


class LatentSpaceSampler:
    def __init__(self, encoder, compute_z_cov=None):
        self.encoder = encoder
        self.multi_output_encoder = False
        if compute_z_cov is None:
            if len(self.encoder.outputs) > 1:
                self.multi_output_encoder = True
        else:
            self.multi_output_encoder = compute_z_cov

        self.z_cov = None

    def get_z_cov(self):
        if self.z_cov is None:
            raise ValueError('get_z_covariance must be called before this function is usable.')
        return self.z_cov

    def get_z_covariance(self, batches_of_xs):
        """Takes one or more batches of xs of shape batches X data_dims"""
        if self.multi_output_encoder:
            # Uncomment the following line if you want to fit multivariate gaussian for sampling for normal vaes too
            # zs = self.encoder.predict(batches_of_xs)[0]

            # Comment the following lines to not use unit variance for sampling all the time for normal VAE
            dim_z = K.get_variable_shape(self.encoder.outputs[0])[-1]
            z_origina_shape = (batches_of_xs.shape[0], dim_z)
            return np.eye(dim_z), z_origina_shape
        else:
            zs = self.encoder.predict(batches_of_xs)
        z_origina_shape = zs.shape
        zs = np.reshape(zs, (z_origina_shape[0], -1))
        self.z_cov = np.cov(zs.T)
        return self.z_cov, z_origina_shape

    def get_zs(self, batches_of_xs=None, z_dim=None, num_smpls=None):
        """batches_of_xs are only used to compute variance of Z on the fly"""
        if batches_of_xs is not None:
            if num_smpls is None:
                num_smpls = batches_of_xs.shape[0]
            if self.encoder is None:
                raise ValueError("No encoder was provided. Can not compute Z variance on the fly")
            self.z_cov, z_origina_shape = self.get_z_covariance(batches_of_xs)
            if z_dim is None:
                z_dim = z_origina_shape
            elif np.prod(z_dim) != self.z_cov.shape[0]:
                raise ValueError("Ambiguous Z-dim. Encoder says its" + str(self.z_cov.shape[0]) +
                                 " Prived " + str(z_dim))
        else:
            if num_smpls is None:
                raise ValueError("Can not infer vslue for num_smpls")
            if z_dim is None:
                raise ValueError("you must provide dimentionality of Z. Can not infer autometically")
            self.z_cov = np.eye(np.prod(z_dim[1:]))
        try:
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ), cov=self.z_cov, size=num_smpls)
        except np.linalg.LinAlgError as e:
            print(self.z_cov)
            print(e)
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ),
                                                         cov=self.z_cov+1e-5*np.eye(self.z_cov.shape[0]),
                                                         size=num_smpls)

        return np.reshape(zs_flattened, (num_smpls,) + z_dim[1:])


class SaveReconstructedImages(keras.callbacks.Callback):
    def __init__(self, epoch_freq, models, test_subset, log_dir, num_samples=None, get_writer_frm=None, log_fid=False,
                 last_epoch=np.inf, num_last_epoch_fid_samples=0):
        self.epoch_freq = epoch_freq
        self.last_epoch = last_epoch
        self.num_last_epoch_fid_samples = num_last_epoch_fid_samples
        self.max_smpls_in_fid_batch = 5
        self.log_fid = log_fid
        self.encoder, self.decoder, self.vae = models
        self.test_subset = test_subset
        self.current_epoch = 0
        if isinstance(test_subset, types.GeneratorType) or hasattr(test_subset, 'next'):
            assert num_samples is not None
            self.generator = True
        else:
            self.generator = False

        if num_samples is None:
            num_samples = test_subset.shape[0]

        self.test_subset = test_subset
        self.num_samples = num_samples
        if get_writer_frm is None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = None
            self.get_writer_from = get_writer_frm
        self.original_img = None
        if self.encoder is not None and self.decoder is not None:
            self.z_sampler = LatentSpaceSampler(self.encoder)

        self.fid_computer = FIDComputer(max_smpls_in_batch=self.max_smpls_in_fid_batch)

        self.original_img = tf.clip_by_value(tf.placeholder(tf.float32), 0, 1.0)
        self.recon_images = tf.clip_by_value(tf.placeholder(tf.float32), 0, 1.0)
        self.recon_fid = tf.placeholder(tf.float32)
        self.smpl_fid = tf.placeholder(tf.float32)
        self.sampled_images = tf.clip_by_value(tf.placeholder(tf.float32), 0, 1.0)

        tf.summary.image("Original_images", self.original_img, max_outputs=16)
        tf.summary.image("Reconstructed_images", self.recon_images, max_outputs=16)
        tf.summary.scalar("Reconstructed_img_FID", self.recon_fid)
        tf.summary.scalar("Smpl_img_FID", self.smpl_fid)
        tf.summary.image("Sampled_images", self.sampled_images, max_outputs=16)
        self.merged_summary = tf.summary.merge_all()

    def _get_random_samples(self, batches_of_xs):
        sampled_images = None
        if self.encoder is not None and self.decoder is not None:
            zs = self.z_sampler.get_zs(batches_of_xs)
            sampled_images = self.decoder.predict(zs)
        return sampled_images

    def get_current_epoch(self):
        return self.current_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_freq == 0:
            self.current_epoch = epoch
            if epoch == self.last_epoch:
                self.num_samples = self.num_last_epoch_fid_samples
            if self.writer is None:
                self.writer = self.get_writer_from.writer
            if self.generator:
                images = None
                original_img = None
                sampled_images = None
                while True:
                    # self.original_img[i*batch_size:(i+1)*batch_size, :] = backup_test_gen.next()[0][0]
                    temp_original = self.test_subset.next()[0]
                    if original_img is None:
                        original_img = temp_original
                        images = self.vae.predict(temp_original)
                        sampled_images = self._get_random_samples(original_img)
                    else:
                        original_img = np.concatenate((original_img, temp_original), axis=0)
                        images = np.concatenate((images, self.vae.predict(temp_original)), axis=0)
                        if sampled_images is not None:
                            sampled_images = np.concatenate((sampled_images, self._get_random_samples(temp_original)),
                                                            axis=0)
                    if original_img.shape[0] >= self.num_samples:
                        break
            else:
                images = self.vae.predict([self.test_subset[0][0][:self.num_samples, :],
                                           self.test_subset[0][1][:self.num_samples, :]])
                original_img = self.test_subset[0][0][:self.num_samples, :]

            original_img = original_img[:self.num_samples]
            images = images[:self.num_samples]
            sampled_images = sampled_images[:self.num_samples]

            if self.log_fid:
                recon_fid = self.fid_computer.get_fid(K.get_session(), original_img, images)
            else:
                recon_fid=0

            if sampled_images is not None:
                if self.log_fid:
                    smpl_fid = self.fid_computer.get_fid(K.get_session(), original_img, sampled_images)
                else:
                    smpl_fid=0

            print("Sampling FID: " + str(smpl_fid) + ", Recon FID: " + str(recon_fid))

            self.writer.add_summary(K.get_session().run(self.merged_summary,
                                                        feed_dict={self.original_img: original_img,
                                                                   self.recon_images: images,
                                                                   self.sampled_images: sampled_images,
                                                                   self.recon_fid: recon_fid,
                                                                   self.smpl_fid: smpl_fid}), epoch)

            self.writer.flush()

    def on_train_end(self, _):
        if self.get_writer_from is None:
            self.writer.close()
