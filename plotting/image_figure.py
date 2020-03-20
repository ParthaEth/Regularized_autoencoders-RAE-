#!/usr/bin/env python
"""
Created on 19.03.2019
@author: Mehdi S. M. Sajjadi
Reads generated images and stacks them together for use in figures for papers.
"""
import os
import argparse
import numpy as np
import glob
import cv2
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(
    description='Image figure creator',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--indir', type=str, required=True,
                    help='directory containing reference images')
parser.add_argument('--outdir', type=str, required=True,
                    help='output directory for final images')
parser.add_argument('--random_seed', type=int, default=11,
                    help='random seed for image selection')
parser.add_argument('--cols', type=int, default=6, help='number of columns')
parser.add_argument('--rows', type=int, default=4, help='number of rows')
args = parser.parse_args()


def loadimg(fn, rgb=True, quantize=False):
    'Loads image from disk.'
    img = cv2.imread(fn, cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def saveimg(img, filename, scale=1):
    'Saves image to disk.'
    img = img.astype(np.float32)
    if np.shape(img)[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = scale*img
    if not cv2.imwrite(filename, img):
        raise ValueError("Could not write image. Access issue. Are u running from a propoer location? "
                         "In case of relative paths.")


def batch(items, n):
    for i in range(0, len(items), n):
        yield items[i: i+n]


def gen_random(path, cols, rows):
    fns = sorted(glob.glob(os.path.join(path, '*.png')))
    np.random.shuffle(fns)
    imgs = batch([loadimg(fn) for fn in fns[:cols*rows]], rows)
    img = np.hstack([np.vstack(row) for row in imgs])
    return img


def gen_reconstruction(origina_data_path, recon_path, cols, rows):
    original_imgs = sorted(glob.glob(os.path.join(origina_data_path, '*.png')))
    recon_imgs = sorted(glob.glob(os.path.join(recon_path, '*.png')))
    shuffle_idx = range(len(original_imgs))
    np.random.shuffle(shuffle_idx)

    recon_imgs = np.array(recon_imgs)[shuffle_idx]
    original_imgs = np.array(original_imgs)[shuffle_idx]

    img_rows = []
    for r in range(0, (rows+2)//2, 2):
        img_row_even = []
        img_row_odd = []
        for c in range(cols):
            img_row_even.append(loadimg(original_imgs[cols*r + c]))
            img_row_odd.append(loadimg(recon_imgs[cols * r + c]))
        img_rows.append(np.hstack(img_row_even))
        img_rows.append(np.hstack(img_row_odd))
    return np.vstack(img_rows)


def gen_interpolation(base_dir_path, cols, rows):
    list_interps = os.listdir(base_dir_path)
    list_interps = sorted(list_interps)
    np.random.shuffle(list_interps)
    interpolation_rows = []
    for row in range(rows):
        img_row = []
        list_interpolations_imgs = sorted(os.listdir(os.path.join(base_dir_path, list_interps[row])))
        if len(list_interpolations_imgs) != cols:
            ValueError("number of images availablein interpolation dir: " +
                       str(os.path.join(base_dir_path, list_interps[row])) + " is not same as requested: " + str(cols))
        for img_name in list_interpolations_imgs:
            img_row.append(loadimg(os.path.join(base_dir_path, list_interps[row], img_name)))
        interpolation_rows.append(np.hstack(img_row))
    interpolation_rows = np.vstack(interpolation_rows)
    return interpolation_rows


def process_directory(indir, outdir, cols, rows, seed):
    np.random.seed(seed)
    indir = os.path.expanduser(indir)
    outdir = os.path.expanduser(outdir)
    outdir = os.path.join(outdir, 'rand_seed'+str(seed))
    os.mkdir(outdir)

    # datasets = ['CIFAR_10', 'MNIST', 'CELEBA']
    datasets = ['MNIST', 'CIFAR_10', 'CELEBA']
    model_names = ['WAE_1', 'VAE_1', 'CV-VAE_1', 'FAE-GP_1', 'FAE-L2_1', 'FAE-SN_1', 'FAE_1', 'AE_1']
    # model_names = ['VAE_1']

    mdl_count = 0
    for datset in datasets:
        dataset_base_dir = os.path.join(outdir, datset)
        os.mkdir(dataset_base_dir)
        for model_name in model_names:
            # # Random Smpls N(0,I)
            np.random.seed(seed)
            try:
                img_random = gen_random(os.path.join(indir, str(mdl_count), model_name, 'one_gaussian_sampled'), cols, rows)
                saveimg(img_random, os.path.join(dataset_base_dir, model_name + '_N_0_I_smpl.png'))
            except Exception as e:
                print(e)

            # Random Smpls N(0,I) 2stage VAE
            np.random.seed(seed)
            try:
                img_random = gen_random(os.path.join(indir, str(mdl_count), model_name, 'aux_vae_sampled'), cols, rows)
                saveimg(img_random, os.path.join(dataset_base_dir, model_name + '_N_0_I_2stg_smpl.png'))
            except Exception as e:
                print(e)

            # # Random Smpls GMM_1
            np.random.seed(seed)
            try:
                img_random = gen_random(os.path.join(indir, str(mdl_count), model_name, 'GMM_1_sampled'), cols, rows)
                saveimg(img_random, os.path.join(dataset_base_dir, model_name + '_GMM_1_sampled.png'))
            except Exception as e:
                print(e)

            # Random Smpls GMM_10
            np.random.seed(seed)
            try:
                img_random = gen_random(os.path.join(indir, str(mdl_count), model_name, 'GMM_10_sampled'), cols, rows)
                saveimg(img_random, os.path.join(dataset_base_dir, model_name + '_GMM_10_sampled.png'))
            except Exception as e:
                print(e)

            # Recon_images
            np.random.seed(seed)
            try:
                img_recon = gen_reconstruction(os.path.join(indir, str(mdl_count), model_name, 'recon_original'),
                                               os.path.join(indir, str(mdl_count), model_name, 'reconstructed'),
                                               cols, rows)
                saveimg(img_recon, os.path.join(dataset_base_dir, model_name + '_recon_images.png'))
            except Exception as e:
                print(e)

            # Recon_images
            np.random.seed(seed)
            try:
                img_recon = gen_reconstruction(os.path.join(indir, str(mdl_count), model_name, 'recon_original'),
                                               os.path.join(indir, str(mdl_count), model_name, 'reconstructed_aux_vae'),
                                               cols, rows)
                saveimg(img_recon, os.path.join(dataset_base_dir, model_name + '_recon_images_aux_vae.png'))
            except Exception as e:
                print(e)

            # Interpolations
            # Linear rescaled
            np.random.seed(seed)
            try:
                interpolation_imgs = gen_interpolation(os.path.join(indir, str(mdl_count), model_name,
                                                                'interpolation_viz/linear_interpolation_re_scaled_viz/'),
                                                       cols, rows)
                saveimg(interpolation_imgs, os.path.join(dataset_base_dir, model_name +
                                                         '_linear_interpolation_re_scaled_images.png'))
            except Exception as e:
                print(e)


            # Linear aux_vae
            np.random.seed(seed)
            try:
                interpolation_imgs = gen_interpolation(os.path.join(indir, str(mdl_count), model_name,
                                                       'interpolation_viz/linear_interpolation_re_scaled_vizaux_vae/'),
                                                       cols, rows)
                saveimg(interpolation_imgs, os.path.join(dataset_base_dir, model_name +
                                                         '_linear_interpolation_re_scaled_images_aux_vae.png'))
            except Exception as e:
                print(e)

            # # Spherical
            # np.random.seed(seed)
            # interpolation_imgs = gen_interpolation(os.path.join(indir, str(mdl_count), model_name,
            #                                                     'interpolation_viz/spherical_interpolation_viz/'),
            #                                        cols, rows)
            # saveimg(interpolation_imgs, os.path.join(dataset_base_dir, model_name +
            #                                          '_spherical_interpolation_images.png'))
            mdl_count += 1


def main():
    process_directory(
        args.indir,
        args.outdir,
        args.cols,
        args.rows,
        args.random_seed)


if __name__ == "__main__":
    main()
