# CLAMPED

## Directory Structure

The primary module code is located inside the `csl` directory.  
  
At the module level:  
`datasets.py` - Adds datasets into the automatic training framework.   
`synthesizers.py` - Adds synthetic models into the automatic training framework.  
`data_generators` - Directory containing definitions of synthetic models that generate data. Contains GANs and VAEs.  
`utils` - Utility functions for visualizations, managing directories, etc.  

Tutorials for running the module code are located in the `notebooks` directory. These notebooks have runnable examples of the synthetic data generation with different datasets, models, and privacy options.  

Research code and current experimentation that hasn't been formally added to the module are located in the `experiments` directory. Particular experiments of interest include:  
`asynch` - Experiments for the collaborative training framework.  
`immediate_sensitivity` - Experiments for diffrential privacy implementations and evaluations.  
`task3` - Experiments for synthetic data generation.  
