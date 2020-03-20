#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg

class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# -------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)

    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        if op.name.find('encoder') >= 0 or op.name.find('decoder') >= 0 or op.name.find('full_VAE') >= 0:
            continue
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


# -------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


# -------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# -------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    # inception_path = pathlib.Path(inception_path)
    model_file = os.path.join(inception_path, 'classify_image_graph_def.pb')
    if not os.path.exists(model_file):
        print("Downloading Inception model")
        import urllib
        import tarfile
        fn, _ = urllib.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', inception_path)
    return str(model_file)


def _handle_path(path, sess):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['m'][:], f['s'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = os.listdir(path)
        x = np.array([imread(os.path.join(path, str(fn))).astype(np.float32) for fn in files[0:10000]])
        if x.shape[-1] != 3:
            x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        m, s = calculate_activation_statistics(x, sess)
    return m, s


def calculate_fid_given_paths(paths, inception_path):
    ''' Calculates the FID of two paths. '''
    # inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess)
        m2, s2 = _handle_path(paths[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


def get_fid(dataset_name, dataset_dir, test_dir):
    inception_path = '/is/ps2/pghosh/datasets/inceptionv1_for_inception_score.pb'
    compare_dataset_name = os.path.join(dataset_dir, dataset_name.upper() + '_STATS.npz')
    if not os.path.exists(compare_dataset_name):
        with tf.Session() as sess:
            create_inception_graph(str(inception_path))
            if os.path.exists(os.path.join(dataset_dir, 'test/test')):
                m, s = _handle_path(os.path.join(dataset_dir, 'test/test'), sess)
            else:
                for datase_files in os.listdir(dataset_dir):
                    if datase_files.endswith('.npz'):
                        break
                x = np.load(os.path.join(dataset_dir, datase_files))['x_test'][:, :, :, 0]
                if x.shape[-1] != 3:
                    x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
                m, s = calculate_activation_statistics(x, sess)
        np.savez(compare_dataset_name, m=m, s=s)
    fid = calculate_fid_given_paths([test_dir, compare_dataset_name], inception_path)
    return fid


if __name__ == "__main__":
    COMPUTE_DATASET_STASTICS = False
    RECONSTRUCTIONS = False

    # dataset = 'celebA'
    # dataset = 'MNIST'
    dataset = 'CIFAR_10'
    if dataset == 'celebA':
        compare_dataset_name = 'celebA_stats.npz'
    elif dataset == 'MNIST':
        compare_dataset_name = 'mnist_stats.npz'
    elif dataset == 'CIFAR_10':
        compare_dataset_name = '/is/ps2/pghosh/datasets/cifar/CIFAR_10_STATS.npz'
    else:
        raise NotImplementedError('statistics fro given dataset is not found')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if COMPUTE_DATASET_STASTICS:
        if dataset == 'MNIST':
            from keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.astype('float32')
            x_train -= 127.0
            x_train /= 127.0
            x_train = np.repeat(np.expand_dims(x_train, axis=-1), 3, axis=-1)
        elif dataset == 'CIFAR_10':
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_test = x_train.astype('float32')
            # x_train -= 127.0
            # x_test /= 255

        with tf.Session() as sess:
            inception_path = check_or_download_inception(None)
            create_inception_graph(str(inception_path))
            if dataset == 'MNIST' or dataset == 'CIFAR_10':
                m, s = calculate_activation_statistics(x_test, sess)
            elif dataset == 'celebA':
                m, s = _handle_path('/is/ps2/pghosh/datasets/celebA64x64/test/test', sess)
            np.savez(compare_dataset_name, m=m, s=s)

    # fid_value = calculate_fid_given_paths(['/is/ps2/pghosh/celebA64x64/test/test', compare_dataset_name], None)
    # print("True data FID: ", fid_value)
    print("True data FID:  2.8198739711940277")

    inception_path = '/is/ps2/pghosh/datasets/inceptionv1_for_inception_score.pb'
    if RECONSTRUCTIONS:
        fid_value = calculate_fid_given_paths(['./generated_samples/celebA/reconstructed/const_H_vae', compare_dataset_name], inception_path)
        print("const_H_vae FID: ", fid_value)

        fid_value = calculate_fid_given_paths(['./generated_samples/celebA/reconstructed/normal_vae', compare_dataset_name], inception_path)
        print("normal_vae FID: ", fid_value)
    else:
        # fid_value = calculate_fid_given_paths(['./generated_samples/celebA/sampled/const_H_vae', compare_dataset_name], None)
        # print("const_H_vae FID: ", fid_value)
        #
        # fid_value = calculate_fid_given_paths(['./generated_samples/celebA/sampled/normal_vae', compare_dataset_name], None)
        # print("normal_vae FID: ", fid_value)
        fid_value = calculate_fid_given_paths(['/is/ps2/pghosh/repos/high_res_vae (3rd copy)/logs/4/gradient_penalty_8/reconstructed', compare_dataset_name], inception_path)
        # fid_value = calculate_fid_given_paths(
        #     ['/is/ps2/pghosh/datasets/cifar/cifat_10/test',
        #      compare_dataset_name], inception_path)
        print("FID: ", fid_value)
