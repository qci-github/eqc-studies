Code and data associated with 

# Zeno Blockade Enabling Photonic Quantum Optimization

by Mohammad-Ali Miri, Uchenna Chukwu, and
Nicholas Chancellor

#### arXiv: XXXXXX #####


#### This code was written by Nicholas Chancellor and written for Quantum Computing Inc. 

#### licenced under [Apache 2.0](http://www.apache.org/licenses/) ####

The code contained here is all written in Python and consists of a jupyter notebook `optical_annealing_paper_plots.ipynb` which creates the plots by calling three custom modules which are also included. It includes the code to produce all of the necessary data, but in cases where the data take a long time to produce, the data is also saved in `.npy` files, which can also be found in the top-level directory. The cells for producing these data are commented out so that the notebook can be run quickly to reproduce the plots. Note that the plots produced here show the same data as the paper, but are not visually identical, as aesthetic improvements were made by Joel Russell Huffman.

### The modules are as follow:

> `QO_trunc_functions.py` this module enables basic state-vector quantum optics simulation using a truncated-state-space approach, it builds upon a module of the same name which was used in [arXiv:2510.19730](https://arxiv.org/abs/2510.19730), but is not identical to it.

> `density_matrix_extension.py` extends functionality to the density matrix formalism, using the same truncated-state-space approach.

> `computing_constructions.py` contains code specifically to implement the computing-related constructions discussed in the manuscript.

### The saved data are as follow:

> `vary_gammat_int_data_ideal_drive.npy` contains data for the plots of success probability versus number of cycles for various coherent levels. These data take **over 24 hours** to produce on the consumer-model laptop used in the studies

> `vary_eta_gamma_data_sfg_drive.npy` contains data for plots of success probability of Zeno blockade with various levels of loss on sum-frequency generation. These data take about **an hour** to run on the consumer-model laptop used in the study.

> `vary_eta_gamma_data_markov_approach.npy` contains data for the plots showing the approach to Markovianty in a Zeno blockade system for various levels of loss. These data take about **six hours** to run on the consumer-model laptop used in the study.

Data which can be produced quickly (less than about 15 minutes) are not stored, but can be recreated using the contained code.


All figures are produced in the `figures` subfolder to keep the top-level directory relatively uncluttered.
