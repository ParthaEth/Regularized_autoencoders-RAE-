# Regularized_autoencoders(RAE)
## This is the official implementation of the Paper titled 'From variational to deterministic Autoencoders'
If you find our work useful please cite us as the following.
```
@inproceedings{
ghosh2020from,
title={From Variational to Deterministic Autoencoders},
author={Partha Ghosh and Mehdi S. M. Sajjadi and Antonio Vergari and Michael Black and Bernhard Scholkopf},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=S1g7tpEYDS}
}
```

## Set up
* Create a virtual environment `virtualenv --no-site-packages <your_home_dir>/.virtualenvs/rae`
* Activate your environment `source <your_home_dir>/.virtualenvs/rae/bin/activate`
* clone the repo `git clone ...`
* Navigate to RAE directory `cd eegularized_autoencoders-rae-` 
* Install requirements `pip install -r requirements.txt`
* Run training `python train_test_var_reduced_vaes.py <config_id> `