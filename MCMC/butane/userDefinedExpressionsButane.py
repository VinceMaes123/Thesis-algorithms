#This file contains all distributions, potentials, gradients... that the user of the mM-MCMC method should provide
#note that in the current version, the restriction should return a scalar
#(only one macro variable; 1D macro description)
#Butane (united atom description without hydrogen molecules) has 6 degrees of freedom:
#3 bond lengths, 2 bond angles and one torsion angle.

#Haal cos,sin,exp... uit autograd.numpy als je autograd op je functie wilt gebruiken, want hierop is autograd gebaseerd!
#als je vb math.cos(..) of scipy.cos(..) gebruikt herkent autograd.grad(func) dit niet!
import scipy
from scipy import linalg
import math
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#############################################################

#Exact expressions for precomputations
#A_hat EXACT (= macro potential A)
def A_hat(eta):
    c0 = 1031.36  # Torsion angle coefficients
    c1 = 2037.82
    c2 = 158.52
    c3 = -3227.7
    A = c0 + c1 * np.cos(eta) + c2 * (np.cos(eta)) ** 2.0 + c3 * (np.cos(eta)) ** 3.0 # Torsion angle energy
    return A

def mu_0(eta,solver):
    return math.exp(-solver.beta*A_hat(eta))

#b EXACT (= -grad(A))
def b_exact(eta):
    dA = grad(A_hat)
    result = -dA(eta)
    return result

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
def restriction(q):
    eta = q[5]
    return eta

#Gradient of restriction
def gradRestriction(q):
    grad = scipy.zeros(6)
    grad[0] = 0.0
    grad[1] = 0.0
    grad[2] = 0.0
    grad[3] = 0.0
    grad[4] = 0.0
    grad[5] = 1.0
    return grad

#Laplacian of restriction
def lapRestriction(q):
    lap = 0.0
    return lap

#Potential V at the micro level
def totalPot(q,solver):
    kb = 1.17e6  # stiffness C-C bond
    r0 = 1.53  # equilibrium C-C bond length
    ka = 62500.0  # stiffness C-C-C bond angle
    theta0 = (112.0 / 360.0) * 2.0 * math.pi  # equilibrium C-C-C bond angle
    c0 = 1031.36  # Torsion angle coefficients
    c1 = 2037.82
    c2 = 158.52
    c3 = -3227.7
    E1 = 0.5 * kb * (q[0] - r0) ** 2  # bond energy 1
    E2 = 0.5 * kb * (q[1] - r0) ** 2  # bond energy 2
    E3 = 0.5 * kb * (q[2] - r0) ** 2  # bond energy 3
    E4 = 0.5 * ka * (q[3] - theta0) ** 2  # bond angle energy 1
    E5 = 0.5 * ka * (q[4] - theta0) ** 2  # bond angle energy 2
    E6 = c0 + c1 * np.cos(q[5]) + c2 * (np.cos(q[5])) ** 2.0 + c3 * (np.cos(q[5])) ** 3.0  # Torsion angle energy
    return E1 + E2 + E3 + E4 + E5 + E6

#Gradient of the potential V at the micro level for stepper
def totalgradPot(q,solver):
    dV = grad(totalPot)
    result = dV(q,solver)
    return result


#############################################################

#other densities we want to plot

#Plotting function
def makePlot(psi,xminPsi,xmaxPsi,func,solver,plotTitleString):
    scaling_Psi = integrate.quad(func, xminPsi, xmaxPsi, args=(solver,))
    scaling_Psi = scaling_Psi[0]
    x_coordinate_Psi = scipy.linspace(xminPsi, xmaxPsi, 1000)
    y_coordinate_Psi = scipy.zeros(scipy.size(x_coordinate_Psi))
    for index in range(0, len(x_coordinate_Psi)):
        y_coordinate_Psi[index] = (1.0 / scaling_Psi) * func(x_coordinate_Psi[index], solver)

    plt.figure()
    plt.hist(psi, bins=1000, range=[xminPsi, xmaxPsi], density=1)  # histogram with 1000 bins
    plt.plot(x_coordinate_Psi, y_coordinate_Psi)
    plt.xlabel(plotTitleString)
    plt.show()
    return 0


#marginal density of a C-C bond length
def CC_0(eta,solver):
    kb = 1.17e6  # stiffness C-C bond
    r0 = 1.53  # equilibrium C-C bond length
    A = 0.5 * kb * (eta - r0) ** 2
    return math.exp(-solver.beta * A)

#marginal density of a C-C-C bond angle
def CCC_0(eta,solver):
    ka = 62500.0  # stiffness C-C-C bond angle
    theta0 = (112.0 / 360.0) * 2.0 * math.pi  # equilibrium C-C-C bond angle
    A = 0.5 * ka * (eta - theta0) ** 2
    return math.exp(-solver.beta * A)

