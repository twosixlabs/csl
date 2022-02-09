# Immediate Sensitivity

This directory includes experiment runners and data analysis for immediate sensitivty experiments.

## Experiments

Each dataset has two files that run experiments: 
 - `[dataset]_experiment.py` trains models with immediate sensitivity.
 - `[dataset]_baseline.py` trains models with the gradient clipping approach to differential privacy.

After these programs have trained models, we save their weights and data to a series of pickle files in the `csl/data/[dataset]` directory.


### Datsets

We run experiments on the Texas-100, Purchase-100, Cifar-10, and FMNIST datasets


## Membership Inference Analysis

A series of python notebooks generate the plots.  The following highlights differences between them

 - `tex_mult_analysis.ipynb` analyses texas-100 models that have been trained several times over to determine their variance
 - `texas_sens_analysis.ipynb` looks at how immediate sensitivity changes during training. 
 - `texas_width_analysis.ipynb` considers how membership inference advantage might change with a larger or smaller model.
 - `purchase_mult_analysis.ipynb` is the same as `tex_mult_analysis.ipynb` but for the Purchase-100 dataset.

 - `pareto.ipynb` generates pareto fronts for all three datasets.

There are a few more notebooks, but these include the primary results.

## Data Poisoning Analysis

The following Python files run the trials and generate the results:

 - `data_poisoning_experiment_baseline.py`
 - `data_poisoning_experiment.py`
 - `data_poisoning_harness_baseline.py`
 - `data_poisoning_harness.py`

The following notebooks calculate epsilon values and graph the results:

 - `Data Poisoning Attack Epsilon Calculation.ipynb`
 - `Data Poisoning Graphs.ipynb`

## Library 

A few python files in this directory comprise a library of helpful functions:

 - `immediate_sensitivity_primitives.py` includes functions that measure immediate sensitivity during training as well as perform gradient clipping.
 - `membership_inference.py` is our implementation of the Yeom and Merlin membership inference attacks.
 - `experiment_runner.py` contains training loops for the immediate sensitivity and graident clipping experiments.


