The attack script attack.py performs the adapted conditional Hayes attack on
conditional MNIST GANs and the standard Hayes attack on unconditional CelebA
WGANs developed by Two Six Technologies as part of DARPA's CSL program.

As stated in the docstring, the attack may be executed with a command like

python -m attack \
--gpu 0 \
--generate_samples \
--compute_fid \
--save \
--data_dir /path/to/data/ \
--model_dir /path/to/models/ \
--model_name MNIST-Baseline \
--checkpoint_min 100 \
--checkpoint_step 100 \
--checkpoint_max 1000 

and should run without code modifications under two assumptions:

1. The code is executed such that the repos csl-gan/ and csl-gan/opacus/
are subdirectories of your current working directory.
2. The --data_dir and --model_dir arguments are given to the directories
*containing* your specific data and model directories. For instance,
your csl-gan MNIST GAN might source data from /path/to/data/MNIST/ 
with model directory /path/to/models/MNIST-Baseline/.