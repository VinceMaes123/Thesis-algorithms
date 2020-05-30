#This file contains all distributions, potentials, gradients... that the user of the mM-MCMC method should provide
#note that in the current version, the restriction should return a scalar
#(only one macro variable; 1D macro description)
#Alanine-dipeptide (united atom description without hydrogen, oxygen molecules and the central subgroup with a carbon)
#has 13 degrees of freedom: 4 C-N bond lengths, 2 C-C bond lengths, 2 torsion angles, 2 C-N-C bond angles and
#3 N-C-C bond angles.
#!!! We only model the main chain of the alanine-dipeptide molecule

#Haal cos,sin,exp... uit autograd.numpy als je autograd op je functie wilt gebruiken, want hierop is autograd gebaseerd!
#als je vb math.cos(..) of scipy.cos(..) gebruikt herkent autograd.grad(func) dit niet!
import scipy
from scipy import linalg
import math
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


###############################################################
#Exact expressions for precomputations
#A_hat EXACT (= macro potential A)
def A_hat(eta):
    kpsi = 2.93e3  # stiffness torsion angle 2 (SLOWEST COMPONENT!)
    A = kpsi * (1 + np.cos(eta + math.pi))  # Torsion angle energy 2
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
    eta = q[12]
    return eta

#Gradient of restriction
def gradRestriction(q):
    grad = scipy.zeros(13)
    grad[0] = 0.0
    grad[1] = 0.0
    grad[2] = 0.0
    grad[3] = 0.0
    grad[4] = 0.0
    grad[5] = 0.0
    grad[6] = 0.0
    grad[7] = 0.0
    grad[8] = 0.0
    grad[9] = 0.0
    grad[10] = 0.0
    grad[11] = 0.0
    grad[12] = 1.0
    return grad

#Laplacian of restriction
def lapRestriction(q):
    lap = 0.0
    return lap

#Potential V at the micro level
def totalPot(q,solver):
    kCC = 1.17e6  # stiffness C-C bond
    rCC = 1.515  # equilibrium C-C bond length
    kCN = 1.147e6  # stiffness C-N bond
    rCN = 1.335 #equilibrium C-N bond length
    kCCN = 2.68e5  # stiffness C-C-N bond angle
    thetaCCN = (113.9 / 360.0) * 2.0 * math.pi  # equilibrium C-C-N bond angle
    kCNC = 1.84e5  # stiffness C-N-C bond angle
    thetaCNC = (117.6 / 360.0) * 2.0 * math.pi  # equilibrium C-N-C bond angle
    kphi = 3.98e4  # stiffness torsion angle 1
    kpsi = 2.93e3  # stiffness torsion angle 2 (SLOWEST COMPONENT!)
    E1 = 0.5 * kCC * (q[0] - rCC) ** 2  # C-C bond energy 1
    E2 = 0.5 * kCC * (q[1] - rCC) ** 2  # C-C bond energy 2
    E3 = 0.5 * kCN * (q[2] - rCN) ** 2  # C-N bond energy 1
    E4 = 0.5 * kCN * (q[3] - rCN) ** 2  # C-N bond energy 2
    E5 = 0.5 * kCN * (q[4] - rCN) ** 2  # C-N bond energy 3
    E6 = 0.5 * kCN * (q[5] - rCN) ** 2  # C-N bond energy 4
    E7 = 0.5 * kCCN * (q[6] - thetaCCN) ** 2  # C-C-N bond angle energy 1
    E8 = 0.5 * kCCN * (q[7] - thetaCCN) ** 2  # C-C-N bond angle energy 2
    E9 = 0.5 * kCCN * (q[8] - thetaCCN) ** 2  # C-C-N bond angle energy 3
    E10 = 0.5 * kCNC * (q[9] - thetaCNC) ** 2  # C-N-C bond angle energy 1
    E11 = 0.5 * kCNC * (q[10] - thetaCNC) ** 2  # C-N-C bond angle energy 2
    E12 = kphi * (1 + np.cos(q[11] + math.pi))  # Torsion angle energy 1
    E13 = kpsi * (1 + np.cos(q[12] + math.pi))  # Torsion angle energy 2
    return E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + E11 + E12 + E13

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

#marginal density of the phi torsion angle
def phi_0(eta,solver):
    kphi = 3.98e4
    A = kphi * (1 + np.cos(eta + math.pi))
    return math.exp(-solver.beta * A)

#marginal density of a C-C bond length
def CC_0(eta,solver):
    kCC = 1.17e6  # stiffness C-C bond
    rCC = 1.515  # equilibrium C-C bond length
    A = 0.5 * kCC * (eta - rCC) ** 2
    return math.exp(-solver.beta * A)

#marginal density of a C-N bond length
def CN_0(eta,solver):
    kCN = 1.147e6  # stiffness C-N bond
    rCN = 1.335  # equilibrium C-N bond length
    A = 0.5 * kCN * (eta - rCN) ** 2
    return math.exp(-solver.beta * A)

#marginal density of a C-C-N bond angle
def CCN_0(eta,solver):
    kCCN = 2.68e5  # stiffness C-C-N bond angle
    thetaCCN = (113.9 / 360.0) * 2.0 * math.pi  # equilibrium C-C-N bond angle
    A = 0.5 * kCCN * (eta - thetaCCN) ** 2
    return math.exp(-solver.beta * A)

#marginal density of a C-N-C bond angle
def CNC_0(eta,solver):
    kCNC = 1.84e5  # stiffness C-N-C bond angle
    thetaCNC = (117.6 / 360.0) * 2.0 * math.pi  # equilibrium C-N-C bond angle
    A = 0.5 * kCNC * (eta - thetaCNC) ** 2
    return math.exp(-solver.beta * A)