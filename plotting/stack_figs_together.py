# Stacks all figures for different models together
from PIL import Image
import os
import numpy as np

datasets = ['MNIST', 'CIFAR_10', 'CELEBA']
model_names = ['VAE_1', 'CV-VAE_1', 'WAE_1', 'VAE_2_stg', 'FAE-GP_1', 'FAE-L2_1', 'FAE-SN_1', 'FAE_1', 'AE_1']
model_nm2_bs_mdl = dict(zip(model_names, model_names))
model_nm2_bs_mdl['VAE_2_stg'] = 'VAE_1'

base_dir = '/home/pghosh/Desktop/fae_iclr/rand_seed11_all'
literals = ['_recon_images.png', '_linear_interpolation_re_scaled_images.png', '_N_0_I_smpl.png']
mld2lit = {}
for key in model_names:
    mld2lit[key] = literals[:]
mld2lit['VAE_2_stg'] = ['_recon_images_aux_vae.png', '_linear_interpolation_re_scaled_images_aux_vae.png', '_N_0_I_2stg_smpl.png']
for i in range(4, len(model_names)):
    mld2lit[model_names[i]][2] = '_GMM_10_sampled.png'

mld2lit['CV-VAE_1'][2] = '_GMM_10_sampled.png'

# Collect literal
for datset in datasets:
    if datset == "CELEBA":
        num_cols = 64 * 6
        num_rows = 64
    else:
        num_cols = 32 * 6
        num_rows = 32

    out_img_file = "recon_" + datset + '.png'

    for lit_idx, curr_lit in enumerate(literals):
        img_stk = []
        for model_name in model_names:
            input_image = np.array(Image.open(
                os.path.join(base_dir, datset, model_nm2_bs_mdl[model_name] + mld2lit[model_name][lit_idx])))

            if model_name == model_names[0]:
                if lit_idx == 0:
                    img_stk.append(input_image[0:2*num_rows, 0:num_cols])
                else:
                    img_stk.append(255 * np.ones(input_image[0:num_rows, 0:num_cols].shape, dtype=np.uint8))
                    img_stk.append(input_image[0:num_rows, 0:num_cols])
            else:
                if lit_idx == 0:
                    img_stk.append(input_image[num_rows:2*num_rows, 0:num_cols])
                else:
                    img_stk.append(input_image[0:num_rows, 0:num_cols])

        final_image = Image.fromarray(np.vstack(img_stk))
        # final_image.resize(size=(len(model_names)+1, 64*6), resample=Image.BICUBIC)
        final_image.save(os.path.join(base_dir, datset + curr_lit))


