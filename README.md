# NGSI Probabilistic Matrix Factorization Code
## Basic Requirements
```bash
    tensorflow == 1.8.0 or tensorflow-gpu == 1.8.0 (recommended)
    matplotlib
    ply
    progressbar
    sklearn
    CUDA compatible GPU x1 (recommended)
```
## Pre-training the neural guider
For pre-training of the neural guider, run:
```bash
    python3 train_guider.py
```
The saved model could be found in ./saved_model/d

## Neurally Guided Structural Inference - Synthetic data

First, use this command to generate synthetic data jobs:
```bash
    python3 guided_synthetic_data.py generate
```
Now you are supposed to find generated jobs in ./data/results.

For a specific item X of the experiment in {"synthetic_1.0_bmf", "synthetic_1.0_gsm", "synthetic_1.0_irm", "synthetic_1.0_mog", "synthetic_1.0_sparse", "synthetic_1.0_bctf", "synthetic_1.0_chain", "synthetic_1.0_ibp", "synthetic_1.0_kf", "synthetic_1.0_pmf"}, run:

```bash
    python3 experiments.py everything X
```

## Neurally Guided Structural Inference - Real World-like Data
We take the image patch analysis task as an example:
### Image Patch
First, use this command to generate a job:
```bash
    python3 image_patch_experiment.py
```

Now you are supposed to find the generated job in ./data/results/image_patch
Then, run 
```bash
    python3 experiments.py everything image_patch
```
to reproduce the results.

# Further Parameter Adjusting

You can find the search settings in ./experiments.py. Also, if you would like to run Roger's version of search, you can replace ./experiments.py with ./experiments_roger.py

# Acknowledgements
Much of the code is borrowed from https://github.com/rgrosse/compositional_structure_search. We thank Roger for his great job!
