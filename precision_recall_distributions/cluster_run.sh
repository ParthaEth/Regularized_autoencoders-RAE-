ARG=$1
source activate opensimrl
# python train_raes_vaes.py $ARG
# python interpolation_fid_and_viz.py $ARG
python generate_nearest_neightbours.py $ARG
