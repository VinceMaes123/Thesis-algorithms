# Thesis-algorithms
Thesis: "micro-Macro acceleration to reduce burn-in time for Markov chain Monte Carlo methods"

Year:2019-2020
Author: Vince Maes

This repository implements the algorithms used in the thesis text:
"micro-Macro acceleration to reduce burn-in time for Markov chain Monte Carlo methods".
These implementations are used to conduct all numerical experiments performed in the thesis.

The folder "HMC" contains implementations of the hybrid Monte Carlo method applied to
-> three-atom molecule
-> test case (simple bivariate Gaussian density)
-> main chain of butane
-> main chain of alanine-dipeptide

The folder "MCMC" contains implementations of the mM-MCMC method and standard MCMC
applied to:
-> three-atom molecule (both mM-MCMC and standard MCMC)
-> butane (only mM-MCMC)
-> alanine-dipeptide (only mM-MCMC)

Needed packages to run the code:
 -> Scipy
 -> Autograd (for butane and alanine-dipeptide implementations)

More info can be found in the "README" files inside the two subfolders and in the comments
in the code itself.
