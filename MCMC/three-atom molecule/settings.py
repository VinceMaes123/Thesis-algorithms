#Solver object contains all parameters needed in the mM-MCMC method
import scipy
import math

class Solver:
    def __init__(self,N=int(1e6),eps=1.0e-6,beta=1.0,x_init=scipy.array([1.0, 0.0, 1.0]),K=5,landa=1.0e6,macroTimestep = 0.01,
                 microTimestep=1.0e-6,zmax=math.pi,zmin=0.0,dz=math.pi/200.0,N_prec=1000,N_gauss=1000):
        # general parameters
        self.N = N  # desired number of samples of the distribution
        self.eps = eps  # time-scale separation
        self.beta = beta  # variable of the micro level dynamics
        self.x_init = x_init
        # stepping parameters
        self.K = K  # number of biased steps taken in indirect reconstruction
        self.landa = landa  # lambda for biased stepping procedure in indirect reconstruction
        self.macroTimestep = macroTimestep  # timestep for macro level
        self.microTimestep = microTimestep  # timestep for micro level
        # precomputation parameters
        self.zmax = zmax  # maximum values for reaction coordinate
        self.zmin = zmin  # minimum values for reaction coordinate
        self.dz = dz  # gridspacing for precomputations (can be a vector)
        self.J = int(math.floor((zmax - zmin) / dz) + 1)  # number of gridpoints for each effective dynamics parameter
        self.N_prec = N_prec  # number of samples taken in precomputations for each z-value in the grid
        self.N_gauss = N_gauss  # number of Gaussian samples taken in normalization constant A_tilde computation
