Boundary-seeking generative adversarial networks (BGAN)

as featured in the paper:
https://arxiv.org/abs/1702.08431v2

.. _email: erroneus@gmail.com

.. _create a GitHub issue: https://github.com/rdevon/BGAN/issues/new

Requirements (rough estimate)
-----------------------------

.. Fuel: http://fuel.readthedocs.io/en/latest/index.html
.. Lasagne: http://lasagne.readthedocs.io/en/latest/
.. Theano (bleeding edge): http://deeplearning.net/software/theano/
.. progressbar2: http://progressbar-2.readthedocs.io/en/latest/

Basic instructions
----------------------

**Note: Very basic. In-depth instuctions forthcoming.**

Datasets are available via Fuel:
http://fuel.readthedocs.io/en/latest/built_in_datasets.html

Install MNIST:

.. code-block:: bash

   $ cd <Dataset directory>

   $ fuel-download binarized_mnist

   $ fuel-convert binarized_mnist

Install CelebA:

.. code-block:: bash

   $ cd <Dataset directory>

   $ fuel-download celeba

   $ fuel-convert celeba 64

Usage
-----
For simple BGAN running on discrete MNIST:

.. code-block:: bash

  python main_discrete.py -o <Output directory` -S <Path to MNIST hdf5>

For simple BGAN running on continuous CelebA:

.. code-block:: bash

  python main_continuous.py -o <Output directory> -S <Path to CelebA hdf5>
 
Basic documentation found in:

.. code-block:: bash

  python main_continuous.py --help

**Note: Published versions of the model are available in the code, and instructions to reproduce will be added soon.**
  
  If there are bugs or clarity is needed to run models, please add to the Issues.
