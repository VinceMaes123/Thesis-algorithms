#This file contains all densities, potentials, gradients... that the user of the mM-MCMC method should provide
#note that in the current version, the restriction should return a scalar
#(only one macro variable; 1D macro description)
import scipy
from scipy import linalg
import math

#THREE-ATOM MOLECULE
#############################################################
#distribution of xa and rc
def distplot(x,epsilon):
    result = math.exp((-0.5) * (x - 1.00) ** 2 / epsilon)
    return result

#Exact expressions for precomputations
#A_hat EXACT (= free energy A)
def A_hat(eta):
    A = (208.0 / 2.0) * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) ** 2
    return A

def mu_0(eta):
    return math.exp(-A_hat(eta))

#b EXACT (= -grad(A))
def b_exact(eta):
    dA = 208.0 * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) * 2 * (eta - (math.pi / 2.0))  # grad(A)
    return -dA

#sigma_exact
def sigma_exact(eta):
    return 1.0

#get Atilde
def getAtilde(eta,solver):
    Atilde = 0.0
    for n in range(0, solver.N_gauss):
        sample = scipy.random.normal(eta, math.sqrt(1 / (solver.landa * solver.beta)))
        Atilde = Atilde + math.exp(-solver.beta * A_hat(sample))
    return Atilde

############################################################
#The real user provided functions
#Restriction
def restriction(x_micro):
    x_c = x_micro[1]
    y_c = x_micro[2]
    eta = scipy.arctan2(y_c , x_c)
    return eta

#Gradient of restriction
def gradRestriction(x_micro):
    x_c = x_micro[1]
    y_c = x_micro[2]
    grad = scipy.zeros(3)
    grad[0] = 0.0
    grad[1] = -y_c/(x_c**2+y_c**2)
    grad[2] = x_c/(x_c**2+y_c**2)
    return grad

#Laplacian of restriction
def lapRestriction(x_micro):
    x_c = x_micro[1]
    y_c = x_micro[2]
    lap = 0.0
    return lap

#Potential V at the micro level
def totalPot(x_micro,solver):
    x_a = x_micro[0]
    x_c = x_micro[1]
    y_c = x_micro[2]
    r_c = math.sqrt(x_c**2 + y_c**2)
    eta = restriction(x_micro)
    V = (1.0 / (2.0 * solver.eps)) * ((x_a - 1.0) ** 2 +
                                        (r_c-1.0) ** 2) + (208.0 / 2.0) * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) ** 2
    return V

#Gradient of the potential V
def totalgradPot(x_micro,solver):
    x_a = x_micro[0]
    x_c = x_micro[1]
    y_c = x_micro[2]
    gradE = gradRestriction(x_micro)
    r_c = math.sqrt(x_c ** 2 + y_c ** 2)
    eta = restriction(x_micro)
    dV = scipy.zeros(3)
    dV[0] = (1.0 / solver.eps) * (x_a - 1.0)
    dVdrc = (1.0 / solver.eps) * (r_c - 1.0)
    dVdeta = 208.0 * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) * 2.0 * (eta - (math.pi / 2.0))
    dV[1] = dVdrc*(x_c/r_c)+dVdeta*gradE[1]
    dV[2] = dVdrc*(y_c/r_c)+dVdeta*gradE[2]
    hxi = scipy.array([0., -x_c/(r_c**2)**1.5, -y_c/(r_c**2)**1.5])
    dV = dV + 1. / (solver.beta * linalg.norm(gradE)) * hxi
    return dV