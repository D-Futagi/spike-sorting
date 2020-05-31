# Spike sorting with misalignment adjustment
## Introduction 
We will propose a code for spike sorting that includes a method for adjusting misalignment. The code is written in Python 3 and developed in OS X 10.15. The currently released code does not include functions that perform automatic parameter optimization. The spike sorting process consists of several functional modules, each of which is described as a member function, and the results of each process are kept in the member variables in modules.py.
## Demonstration
We show an example of spike sorting using simulated data. The demostration code (demo.ipynb) is in Jupyter notebook format and can be browsed in https://github.com/D-Futagi/spike-sorting/blob/master/demo.ipynb. In the code, the process for spike sorting is carried out step by step. The simulated data in this repository was produced by Qurigoa et al. and is available in sitespike.g-node.org. Details of the data can be found in: 

Quiroga, R. Q., Nadasdy, Z. & Ben-Shaul, Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural Comput. 16(8), 1661-1687 (2004).

## Notification
This repository is under construction.
The code for spike sorting in this repository still only support the simulated data proposed by Quiroga et al. 

