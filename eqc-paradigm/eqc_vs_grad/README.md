# Entropy Computing vs Gradient Descent

This repository includes data for experiments involving comparisons of entropy computing and gradient descent as presented in paper
[Entropy Computing: A Paradigm for Optimization in an Open Quantum System](https://arxiv.org/abs/2407.04512)
The data used in the scripts was developed using an experimental version interface to the entropy quantum computing system which is not available to the public.
The results for these runs are shared as zip files within the data directory. 
The requirements file provides details on python version as well as package version that were. Additionally,
the original script that was used to produce the plots is provided. As a final note the data is stored as a zip files to reduce the memory footprint
form the runs and allow them to be  stored in github. The script will unzip the results files for eqc results and then generate
three plots in order to continue running the script simply exit the plots as they pop up they are also saved in the figures directory
as svg formatted images when the script is run.

### Contents
- QPLIB_0018_OBJ.csv: qplib 18 problem that was used for the comparison of grad and eqc
- data/results_*.zip: zipped version of results from eqc there are 5 numbered files of this type
- grad_vs_eqc.py: script to generate all comparison plots for grad vs eqc
- requirements.txt: includes package versions for all python packages to run grad_vs_eqc.py as well as the python version that was used

Questions comments and concerns can be addressed to the authors of the paper see arxiv link.
