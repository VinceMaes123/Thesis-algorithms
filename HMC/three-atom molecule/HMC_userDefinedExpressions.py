#This file contains all distributions, potentials, gradients... that the user of the HMC method should provide

import scipy
from scipy import linalg
import math

#x_micro = q in the HMC method!!!

########################################################################################################################
############################################# 3 ATOM MOLECULE FUNCTIONS ################################################


#Epsilon from 3-atom molecule problem
def epsilon():
    return 1.0e-6

#Beta from 3-atom molecule problem
def beta():
    return 1.0

#Potential V at the micro level
def V(x_micro):
    x_a = x_micro[0]
    x_c = x_micro[1]
    y_c = x_micro[2]
    r_c = math.sqrt(x_c**2 + y_c**2)
    eta = restriction(x_micro)
    V = (1.0 / (2.0 * epsilon())) * ((x_a - 1.0) ** 2 +
                                        (r_c-1.0) ** 2) + (208.0 / 2.0) * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) ** 2
    return V

################################
# FUNCTIONS TO CALCULATE gradV #
################################
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

#Gradient of the potential V at the micro level for stepper
def gradV(x_micro):
    x_a = x_micro[0]
    x_c = x_micro[1]
    y_c = x_micro[2]
    gradE = gradRestriction(x_micro)
    r_c = math.sqrt(x_c ** 2 + y_c ** 2)
    eta = restriction(x_micro)
    dV = scipy.zeros(3)
    dV[0] = (1.0 / epsilon()) * (x_a - 1.0)
    dVdrc = (1.0 / epsilon()) * (r_c - 1.0)
    dVdeta = 208.0 * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) * 2.0 * (eta - (math.pi / 2.0))
    dV[1] = dVdrc*(x_c/r_c)+dVdeta*gradE[1]
    dV[2] = dVdrc*(y_c/r_c)+dVdeta*gradE[2]
    hxi = scipy.array([0., -x_c/(r_c**2)**1.5, -y_c/(r_c**2)**1.5])
    dV = dV + 1. / (beta() * linalg.norm(gradE)) * hxi
    return dV


########################################################################################################################
#distribution of xa and rc
def distplot(x,epsilon):
    result = math.exp((-0.5) * (x - 1.00) ** 2 / epsilon)
    return result

#Exact expressions for precomputations
#A_hat EXACT (= macro potential A)
def A_hat(eta):
    A = (208.0 / 2.0) * ((eta - (math.pi / 2.0)) ** 2 - 0.3838 ** 2) ** 2
    return A

def mu_0(eta):
    return math.exp(-A_hat(eta))