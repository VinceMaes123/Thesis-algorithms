This folder contains the implementation of the two MCMC methods discussed in the thesis.

->"Standard MCMC": MCMC where proposal moves are based on a time discretization of the 
  overdamped Langevin dynamics underlying the invariant density of interest. (In the
  literature, this method is also referred to as the Metropolis Adjusted Langevin
  Algorithm (MALA).)

  This method is only implemented for the three-atom molecule example.

->"mM-MCMC": this method is implemented for all three examples: the three-atom molecule,
  the main chain of butane and the main chain of alanine-dipeptide.

Each subfolder contains the implementation for one of the examples and contains:

1) An executable file
--------------------------------------------------------------------------------------------
  -> "main.py" for the mM-MCMC method applied to the three-atom molecule.
  -> "classicMCMC.py" for the standard MCMC method applied to the three-atom molecule.
  -> "butane.py" for the mM-MCMC method applied to the main chain of butane.
  -> "alaninedipeptide.py" for the mM-MCMC method applied to the main chain of
     alanine-dipeptide.

2) A set of helper files that contain functions used in the algorithms.
--------------------------------------------------------------------------------------------

acceptReject.py
-> Contains all functions related to the accept-reject criterions.
   The implementation is the same in each subfolder!

precomputations.py
-> Contains functions related to precomputations of free energy and effective dynamics
   coefficients; creating a grid for the reaction coordinate; interpolating functions
   between grid values of the reaction coordinate grid.
   (function for precomputing [b,sigma,A] may still contain bugs!)
   The implementation is the same in each subfolder!

settings.py
-> Generate a 'solver object' that contains all relevant parameters for the mM-MCMC
   algorithm.
   The implementation is the same in each subfolder!

stepper.py
-> Contains implementations of all Markov chains used to generate proposal moves.
   The implementation is the same in each subfolder!

userDefinedExpressions.py
-> Contains problem specific functions (potential energy etc.).
   This is the only helper file that is different in each subfolder, because it contains
   problem specific information.

