#This file contains all distributions, potentials, gradients... that the user of the HMC method should provide
#Alanine-dipeptide (united atom description without hydrogen, oxygen molecules and the central subgroup with a carbon)
#has 13 degrees of freedom: 4 C-N bond lengths, 2 C-C bond lengths, 2 torsion angles, 2 C-N-C bond angles and
#3 C-C-N bond angles.
#!!! We only model the main chain of the alanine-dipeptide molecule

#Haal cos,sin,exp... uit autograd.numpy als je autograd op je functie wilt gebruiken, want hierop is autograd gebaseerd!
#als je vb math.cos(..) of scipy.cos(..) gebruikt herkent autograd.grad(func) dit niet!
from autograd import grad
import autograd.numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

#x_micro = q in the HMC method!!!

########################################################################################################################
################################################# ALANINE_DIPEPTIDE ####################################################

def V(q):
    beta = 1.0/100.0 #inverse temperature
    kCC = 1.17e6 #stiffness C-C bond
    rCC = 1.515 #equilibrium C-C bond length
    kCN = 1.147e6 #stiffness C-N bond
    rCN = 1.335
    kCCN = 2.68e5 #stiffness C-C-N bond angle
    thetaCCN = (113.9 / 360.0) * 2.0 * math.pi  # equilibrium C-C-N bond angle
    kCNC = 1.84e5 #stiffness C-N-C bond angle
    thetaCNC = (117.6 / 360.0) * 2.0 * math.pi  # equilibrium C-N-C bond angle
    kphi = 3.98e4 #stiffness torsion angle 1
    kpsi = 2.93e3 #stiffness torsion angle 2 (SLOWEST COMPONENT!)
    E1 = 0.5*kCC*(q[0]-rCC)**2 # C-C bond energy 1
    E2 = 0.5*kCC*(q[1]-rCC)**2 # C-C bond energy 2
    E3 = 0.5*kCN*(q[2]-rCN)**2 # C-N bond energy 1
    E4 = 0.5*kCN*(q[3]-rCN)**2 # C-N bond energy 2
    E5 = 0.5*kCN*(q[4]-rCN)**2 # C-N bond energy 3
    E6 = 0.5*kCN*(q[5]-rCN)**2 # C-N bond energy 4
    E7 = 0.5*kCCN*(q[6]-thetaCCN)**2 # C-C-N bond angle energy 1
    E8 = 0.5*kCCN*(q[7]-thetaCCN)**2 # C-C-N bond angle energy 2
    E9 = 0.5*kCCN*(q[8]-thetaCCN)**2 # C-C-N bond angle energy 3
    E10 = 0.5*kCNC*(q[9]-thetaCNC)**2 # C-N-C bond angle energy 1
    E11 = 0.5*kCNC*(q[10]-thetaCNC)**2 # C-N-C bond angle energy 2
    E12 = kphi*(1+np.cos(q[11]+math.pi)) # Torsion angle energy 1
    E13 = kpsi*(1+np.cos(q[12]+math.pi)) # Torsion angle energy 2
    return beta*(E1+E2+E3+E4+E5+E6+E7+E8+E9+E10+E11+E12+E13)

def gradV(q):
    dV = grad(V)
    result = dV(q)
    return result


#Exact expressions for reaction coordinate
#A_hat EXACT (= macro potential A)
def A_hat(eta):
    beta = 1.0/100.0 #inverse temperature
    kpsi = 2.93e3 #stiffness torsion angle 2 (SLOWEST COMPONENT!)
    A = beta*(kpsi*(1+np.cos(eta+math.pi))) #Torsion angle energy 2
    return A

def mu_0(eta):
    return math.exp(-A_hat(eta))


#############################################################

#other densities we want to plot

#Plotting function
def makePlot(psi,xminPsi,xmaxPsi,func,plotTitleString):
    scaling_Psi = integrate.quad(func, xminPsi, xmaxPsi)
    scaling_Psi = scaling_Psi[0]
    x_coordinate_Psi = scipy.linspace(xminPsi, xmaxPsi, 1000)
    y_coordinate_Psi = scipy.zeros(scipy.size(x_coordinate_Psi))
    for index in range(0, len(x_coordinate_Psi)):
        y_coordinate_Psi[index] = (1.0 / scaling_Psi) * func(x_coordinate_Psi[index])

    plt.figure()
    plt.hist(psi, bins=1000, range=[xminPsi, xmaxPsi], density=1)  # histogram with 1000 bins
    plt.plot(x_coordinate_Psi, y_coordinate_Psi)
    plt.xlabel(plotTitleString)
    plt.show()
    return 0

#marginal density of the phi torsion angle
def phi_0(eta):
    beta = 1.0 / 100.0  # inverse temperature
    kphi = 3.98e4
    A = kphi * (1 + np.cos(eta + math.pi))
    return math.exp(-beta * A)

#marginal density of a C-C bond length
def CC_0(eta):
    beta = 1.0 / 100.0  # inverse temperature
    kCC = 1.17e6  # stiffness C-C bond
    rCC = 1.515  # equilibrium C-C bond length
    A = 0.5 * kCC * (eta - rCC) ** 2
    return math.exp(-beta * A)

#marginal density of a C-N bond length
def CN_0(eta):
    beta = 1.0 / 100.0  # inverse temperature
    kCN = 1.147e6  # stiffness C-N bond
    rCN = 1.335  # equilibrium C-N bond length
    A = 0.5 * kCN * (eta - rCN) ** 2
    return math.exp(-beta * A)

#marginal density of a C-C-N bond angle
def CCN_0(eta):
    beta = 1.0 / 100.0  # inverse temperature
    kCCN = 2.68e5  # stiffness C-C-N bond angle
    thetaCCN = (113.9 / 360.0) * 2.0 * math.pi  # equilibrium C-C-N bond angle
    A = 0.5 * kCCN * (eta - thetaCCN) ** 2
    return math.exp(-beta * A)

#marginal density of a C-N-C bond angle
def CNC_0(eta):
    beta = 1.0 / 100.0  # inverse temperature
    kCNC = 1.84e5  # stiffness C-N-C bond angle
    thetaCNC = (117.6 / 360.0) * 2.0 * math.pi  # equilibrium C-N-C bond angle
    A = 0.5 * kCNC * (eta - thetaCNC) ** 2
    return math.exp(-beta * A)
