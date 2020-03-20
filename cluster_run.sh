ARG=$1
conda activate opensimrl
# python train_raes_vaes.py $ARG
python interpolation_fid_and_viz.py $ARG
