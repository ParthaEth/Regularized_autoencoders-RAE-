# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import hashlib

import numpy as np
import inception
import prd_score as prd


def generate_inception_embedding(imgs, inception_path, layer_name='pool_3:0'):
    return inception.embed_images_in_inception(imgs, inception_path, layer_name)


def load_or_generate_inception_embedding(directory, cache_dir, inception_path):
    hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    if os.path.exists(path):
        embeddings = np.load(path)
        return embeddings
    imgs = load_images_from_dir(directory)
    embeddings = generate_inception_embedding(imgs, inception_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    return embeddings


def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    paths = [os.path.join(directory, fn) for fn in os.listdir(directory)
             if os.path.splitext(fn)[-1][1:] in types]
    # images are in [0, 255]
    imgs = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for path in paths]
    return np.array(imgs)


def compute_prd(reference_dir, eval_dirs, inception_path):
    real_embeddings = load_or_generate_inception_embedding(reference_dir, '/tmp/prd_cache/', inception_path)
    prd_data = []
    for directory in eval_dirs:
    	print('computing inception embeddings for ' + directory)
        eval_embeddings = load_or_generate_inception_embedding(
            directory, '/tmp/prd_cache/', inception_path)
    	print('computing PRD')
        prd_data.append(prd.compute_prd_from_embedding(
            eval_data=eval_embeddings,
            ref_data=real_embeddings,
            num_clusters=20,
            num_angles=1001,
            num_runs=10))
    f_beta_data = [prd.prd_to_max_f_beta_pair(precision, recall, beta=8)
                   for precision, recall in prd_data]
    prd.plot(prd_data)
    print('F_8   F_1/8     model')
    for directory, f_beta in zip(eval_dirs, f_beta_data):
        print('%.3f %.3f     %s' % (f_beta[0], f_beta[1], directory))

