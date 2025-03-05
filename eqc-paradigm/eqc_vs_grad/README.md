# Entropy Computing vs Gradient Descent

This repository includes data for experiments involving comparisons of entropy computing and gradient descent as presented in paper
[Entropy Computing: A Paradigm for Optimization in an Open Quantum System](https://arxiv.org/abs/2407.04512)
The data used in the scripts was developed using an experimental version of a entropy quantum computing system and is not released.
Instead, a zip of the results file *results_500_qplib18.zip* is provided as well as the implementation of gradient that was used to
generate the plots in the paper. The requirements file provides details on python version as well as package version that were. Additionally,
the original script that was used to produce the plot is provided. As a final note the data is stored as a zip file to reduce the memory footprint
it uses in get simpy unzip the file and then if it is isn't already in this directory place the json file in this directory. The script will generate
three plots in order to continue running the script simply exit the plots as they pop up  they are also saved in local directory
as svg formatted images if want to look at after running the script.

### Contents
- QPLIB_0018_OBJ.csv: qplib 18 problem that was used for the comparison of grad and eqc
- results_5000_qplib.zip: zipped version of results from eqc
- grad_vs_eqc.py: script to generate all comparison plots for grad vs eqc
- requirements.txt: includes package versions for all python packages to run grad_vs_eqc.py as well as the python version that was used

Questions comments and concerns can be addressed to the authors of the paper see arxiv link.