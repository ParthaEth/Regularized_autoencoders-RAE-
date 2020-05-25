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
## Watch a brief presentation
[![Watch a presentation](images/presentation_vid.png)](https://www.youtube.com/embed/TiIuFt1KvJ4)

## Full pdf
You can download a full PDF version of our paper from [https://openreview.net/forum?id=S1g7tpEYDS](https://openreview.net/forum?id=S1g7tpEYDS)

## Set up
* Create a virtual environment `virtualenv --no-site-packages <your_home_dir>/.virtualenvs/rae`
* Activate your environment `source <your_home_dir>/.virtualenvs/rae/bin/activate`
* clone the repo `git clone ...`
* Navigate to RAE directory `cd Regularized_autoencoders-RAE-` 
* Install requirements `pip install -r requirements.txt`
* Run training `python train_test_var_reduced_vaes.py <config_id> `

## Data and pretrained models
###celebA
* Please download the [celeba dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* Please pre process as desribed in the original paper. Centre crop `140X140` and resize to `64X64`
* Place it under desired directory and specify the location in the `dataloader.py`
file `line 144`
* the directory structure underneath must be as follows
``` python
celebA_root
    |____ train
            |___ train
                  |___ 0.png
                       1.png
                        .
                        .
                        .
    |____ test
            |___ test
                  |___ 182638.png
                       182639.png
                        .
                        .
                        .
    |____ val
            |___ val
                  |___ 162771.png
                       162772.png
                        .
                        .
                        .
```
### CIFAR
* for CIFAR10 please prepare a `.npz` file `cifar_10.npz`
* this file should contain a numpy array of size `tran_smaplesX32X32X3` under the key 
`x_train`
* and another numpy array of test samples of dimension `10000X32X32X3`
* Values must be in the range `0-255`
* please modify the path to root directory in `line 143` in file `dataloader.py`

### MNIST
* for MNIST please prepare a `.npz` file `mnist_32x32.npz`
* this file should contain a numpy array of size `tran_smaplesX32X32X1` under the key 
`x_train`
* values must be in the range `0-255`
* padd 2 columns and rows of zeros on all four sides to get the dimensionality of MNIST 
samples become `32X32X1`
* modify the root directory path in `line 125` in file `dataloader.py`

### Checkpoints
* Please download the `logs` directory from 
[this dropbox link](https://www.dropbox.com/sh/btz9ctt4zpabs4a/AAARKArjTnsFwqL17rcbsUcba?dl=0)
* specify the location to this directory in the `config.py` file
* and you are all set up to run

### Misc
#### Example configurations
It is a big dictionar with primary and secondary entries. Primary entry holds all 
the common entries while the secondary entry is run specific entries. the run inde is 
simply the flat index number from top. So in the following example to run the setting 
under dictionary key `1` one must run with `run_id` `2` wile `run_id 0` runs the first 
entry under dictionary key `0`. All the results are generated in the `log` directory
under sme directory hierarchy. 
```Python
        configurations = \
            {0: [{'base_model_name': "rae"},
                                       {'expt_name': 'l2_regularization'},

                                       {'expt_name': 'l2_regularization'}
                   ],

             1: [{'base_model_name': "rae"},
                                       {'expt_name': 'spectral_normalization'}
                   ],

             2: [{'base_model_name': "rae"},
                                       {'expt_name': 'l2_regularization'},

                                       {'expt_name': 'l2_regularization'}
                   ],
            }
```
