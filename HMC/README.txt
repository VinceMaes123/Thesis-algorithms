This folder contains the implementation of the HMC method discussed in the thesis.

Each subfolder contains the implementation for one of the examples and contains:

1) An executable file
--------------------------------------------------------------------------------------------
  -> "HMC_main.py" for the HMC method applied to the three-atom molecule.
  -> "HMC_TestCase.py" for the HMC method applied to the test case (simple bivariate Gaussian density)
  -> "HMC_Butane.py" for the HMC method applied to the main chain of butane.
  -> "HMC_AlanineDipeptide.py" for the mM-MCMC method applied to the main chain of
     alanine-dipeptide.

2) A set of helper files that contain functions used in the algorithm.
--------------------------------------------------------------------------------------------

HMC_stepper.py
-> Contains the implementation of the HMC method.
   The implementation is the same in each subfolder!

HMC_userDefinedExpressions.py
-> Contains problem specific functions (potential energy etc.).

