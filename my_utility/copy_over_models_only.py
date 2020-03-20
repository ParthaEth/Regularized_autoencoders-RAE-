# Only runs in python 3
import os
import shutil
import glob
import tqdm

source_root = '/is/cluster/work/pghosh/high_res_vae/logs'
dest_root = '/home/pghosh/Dropbox/RAE_model_checkpoints/logs'

for runid_major in tqdm.tqdm(range(31)):
    run_dir = os.path.join(source_root, str(runid_major))
    for run_dirs in os.listdir(run_dir):
        dir_current_run = os.path.join(run_dir, run_dirs)
        dest_run_dirs = os.path.join(dest_root, str(runid_major), run_dirs)
        os.makedirs(dest_run_dirs, exist_ok=True)
        files_to_be_coppied = glob.glob(os.path.join(dir_current_run, '*.pkl'))
        files_to_be_coppied += glob.glob(os.path.join(dir_current_run, '*.npz'))
        files_to_be_coppied += glob.glob(os.path.join(dir_current_run, '*.h5*'))
        for fl_cpy in files_to_be_coppied:
            shutil.copyfile(fl_cpy, os.path.join(dest_run_dirs, os.path.basename(fl_cpy)))

# for runid_major in tqdm.tqdm(range(31)):
#     run_dir = os.path.join(dest_root, str(runid_major))
#     for run_dirs in os.listdir(run_dir):
#         dir_current_run = os.path.join(run_dir, run_dirs)
#
#         files_to_be_coppied = glob.glob(os.path.join(dir_current_run, '*variance_reduced_vae*'))
#         for fl_cpy in files_to_be_coppied:
#             os.rename(fl_cpy, fl_cpy.replace('variance_reduced_vae', 'RAE'))
