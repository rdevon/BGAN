# BGAN
boundary-seeking generative adversarial networks (BGAN) as featured in the paper:
https://arxiv.org/abs/1702.08431v2

README UNDER CONSTRUCTION...

Basic instructions (in-depth instuctions forthcoming)

Requirements (rough estimate):
Fuel
Lasagne
Theano (bleeding edge)
progressbar2

Datasets are available via Fuel:
http://fuel.readthedocs.io/en/latest/built_in_datasets.html

Usages
For simple BGAN running on discrete MNIST:
python main_discrete.py -o <Output directory> -S <Path to MNIST hdf5>

For simple BGAN running on continuous CelebA:
python main_continuous.py -o <Output directory> -S <Path to CelebA hdf5>

Published versions of the model are available in the code, and instructions to reproduce will be added soon.

If there are bugs or clarity is needed to run models, please add to the Issues.
