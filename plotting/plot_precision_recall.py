import numpy as np
from precision_recall_distributions import prd_score as prd
import os


# labels = ['WAE', 'WAE-GMM', 'RAE-SN']
# paths = ['/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/16/WAE_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/21/FAE-SN_1/prd_data.npz']
#
# prd_wae = np.load(paths[0])['prd_data']
# prd_rae_sn = np.load(paths[1])['prd_data']
#
# prd.plot([prd_wae[0], prd_wae[1], prd_rae_sn[1]], labels=labels, legend_loc='upper_right')


# ALL RAEs
# labels = ['RAE-L2', 'RAE-SN', 'RAE-GP']
# paths = ['/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/20/FAE-L2_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/21/FAE-SN_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/19/FAE-GP_1/prd_data.npz']
#
# prd_rae_l2 = np.load(paths[0])['prd_data']
# prd_rae_sn = np.load(paths[1])['prd_data']
# prd_rae_gp = np.load(paths[2])['prd_data']
#
# prd.plot([prd_rae_l2[1], prd_rae_sn[1], prd_rae_gp[1]], labels=labels, legend_loc='upper_right')

# # All traditional
# labels = ['VAE', 'CV-VAE', 'WAE', 'AE']
# paths = ['/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/17/VAE_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/18/CV-VAE_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/16/WAE_1/prd_data.npz',
#          '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/23/AE_1/prd_data.npz']
#
# prd_vae = np.load(paths[0])['prd_data']
# prd_cv_vae = np.load(paths[1])['prd_data']
# prd_wae = np.load(paths[2])['prd_data']
# prd_ae = np.load(paths[3])['prd_data']
#
# prd.plot([prd_vae[0], prd_cv_vae[1], prd_wae[0], prd_ae[1]], labels=labels, legend_loc='upper right')

# All models GMM VS N(0,1)
base_path = '/ps/scratch/pghosh/frozenModelsICCV_full/high_res_vae/logs/'
models = ['WAE', 'VAE', 'CV-VAE', 'FAE-GP', 'FAE-L2', 'FAE-SN', 'FAE', 'AE']
datasets = ['MNIST', 'CIFAR', 'CELEBA']
store_root = './'

count = 0
for dataset in datasets:
    for model in models:
        store_dir = os.path.join(store_root, dataset)
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        path_prd = os.path.join(base_path, str(count), model+'_1', 'prd_data.npz')
        prd_current_mdl = np.load(path_prd)['prd_data']

        if model in ['CV-VAE', 'RAE', 'AE']:
            base_prd_idx = 1
            labels = ['N(\mu, \sigma)', 'GMM_10']
        else:
            base_prd_idx = 0
            labels = ['N(0, I)', 'GMM_10']

        file_name = os.path.join(store_dir, model+'.png')
        prd.plot([prd_current_mdl[0], prd_current_mdl[2]], labels=labels, legend_loc='upper right', out_path=file_name)
        count += 1
        print(count)

