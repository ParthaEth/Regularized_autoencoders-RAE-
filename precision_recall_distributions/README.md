# Assessing Generative Models via Precision and Recall

Official code for [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035) by [Mehdi S. M. Sajjadi](http://msajjadi.com), [Olivier Bachem](http://olivierbachem.ch/), [Mario Lucic](https://ai.google/research/people/MarioLucic), [Olivier Bousquet](https://ai.google/research/people/OlivierBousquet), and [Sylvain Gelly](https://ai.google/research/people/SylvainGelly), presented at [NeurIPS 2018](https://neurips.cc/). The poster can be downloaded [here](https://owncloud.tuebingen.mpg.de/index.php/s/QbztGoa88zJ8M3a).

## Usage
### Requirements
A list of required packages is provided in [requirements.txt](requirements.txt) and may be installed by running:
```shell
pip install -r requirements.txt
```

If the embedding is computed manually, a minimal set of required packages may be used, see [requirements_minimal.txt](requirements_minimal.txt).

### Automatic: Compute PRD for folders of images on disk
_Note that a GPU will significantly speed up the computation of the Inception embeddings, consider installing `pip install tensorflow-gpu`._

Example: you have a folder of images from your true distribution (e.g., `~/real_images/`) and any number of folders of generated images (e.g., `~/generated_images_1/` and `~/generated_images_2/`). Note that the number of images in each folder needs to be the same.

1. Download the pre-trained inception network from [here](https://owncloud.tuebingen.mpg.de/index.php/s/ef7QgkaX544nzcZ) and place it somewhere, e.g. `/tmp/prd_cache/inception.pb` (_Alternate link [here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz). Note that this file needs to be unpacked._)
2. In a shell, cd to the repository directory and run
```shell
python prd_from_image_folders.py --inception_path /tmp/prd_cache/inception.pb --reference_dir ~/real_images/ --eval_dirs ~/generated_images_1/ ~/generated_images_2/ --eval_labels model_1 model_2
```

For further customization, run `./prd_from_image_folders.py -h` to see the list of available options.

### Manual: Compute PRD from any embedding
Example: you want to compare the precision and recall of a pair of generative models in some feature embedding to your liking (e.g., Inception activations).

1. Take your test dataset and generate the same number of data points from each of your generative models to be evaluated.
2. Compute feature embeddings of both real and generated datasets, e.g. `feats_real`, `feats_gen_1` and `feats_gen_2` as numpy arrays each of shape `[number_of_data_points, feature_dimensions]`.
3. In python, run the following code:
```python
import prd
prd_data_1 = prd.compute_prd_from_embedding(feats_real, feats_gen_1)
prd_data_2 = prd.compute_prd_from_embedding(feats_real, feats_gen_2)
prd.plot([prd_data_1, prd_data_2], ['model_1', 'model_2'])
```

## BibTex citation
```
@inproceedings{precision_recall_distributions,
  title     = {{Assessing Generative Models via Precision and Recall}},
  author    = {Sajjadi, Mehdi~S.~M. and Bachem, Olivier and Lu{\v c}i{\'c}, Mario and Bousquet, Olivier and Gelly, Sylvain},
  booktitle = {{Advances in Neural Information Processing Systems (NeurIPS)}},
  year      = {2018}}
```

## Further information
External copyright for: [prd_score.py](https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score.py) [prd_score_test.py](https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score_test.py)
[inception_network.py](https://github.com/google/compare_gan/blob/master/compare_gan/src/fid_score.py)<br>
Copyright for remaining files: [Mehdi S. M. Sajjadi](http://msajjadi.com)<br>

License for all files: [Apache License 2.0](LICENSE)

For any questions, comments or help to get it to run, please don't hesitate to mail us: <msajjadi@tue.mpg.de>
