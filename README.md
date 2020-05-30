# Thesis-algorithms
Thesis: "micro-Macro acceleration to reduce burn-in time for Markov chain Monte Carlo methods"<br/>
<br/>
Year:2019-2020<br/>
Author: Vince Maes<br/>
<br/>
This repository implements the algorithms used in the thesis text:
"micro-Macro acceleration to reduce burn-in time for Markov chain Monte Carlo methods" in Python.
These implementations are used to conduct all numerical experiments performed in the thesis.<br/>
<br/>
The folder "HMC" contains implementations of the hybrid Monte Carlo method applied to<br/>
-> three-atom molecule<br/>
-> test case (simple bivariate Gaussian density)<br/>
-> main chain of butane<br/>
-> main chain of alanine-dipeptide<br/>
<br/>
The folder "MCMC" contains implementations of the mM-MCMC method and standard MCMC
applied to:<br/>
-> three-atom molecule (both mM-MCMC and standard MCMC)<br/>
-> butane (only mM-MCMC)<br/>
-> alanine-dipeptide (only mM-MCMC)<br/>
<br/>
Special packages needed to run the code:<br/>
 -> Autograd (for butane and alanine-dipeptide implementations)<br/>
<br/>
More info can be found in the "README" files inside the two subfolders and in the comments
in the code itself.
